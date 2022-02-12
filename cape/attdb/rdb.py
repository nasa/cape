#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.attdb.rdb`: Template ATTDB database
========================================================

This module provides the class :class:`DataKit` as a subclass of
:class:`dict` that contains methods common to each of the other database
classes.  The :class:`DataKit` class provides an interface to both store
the data and create and call "response surfaces" that define specific,
potentially complex interpolation methods to evaluate the data as a
function of several independent variables.

Finally, having this common template class provides a single point of
entry for testing if an object is based on a product of the
:mod:`cape.attdb.rdb` module.  The following Python sample tests if
any Python object *db* is an instance of any class from this data-file
collection.

    .. code-block:: python

        isinstance(db, cape.attdb.rdb.DataKit)

This class is the basic data container for ATTDB databases and has
interfaces to several different file types.

"""

# Standard library modules
import copy
import os
import re
import sys

# Third-party modules
import numpy as np

# Semi-optional third-party modules
try:
    import scipy.interpolate as sciint
    import scipy.interpolate.rbf as scirbf
except ImportError:
    sciint = None
    scirbf = None

# Local modules
from . import ftypes
from ..tnakit import kwutils as kwutils
from ..tnakit import plot_mpl as pmpl
from ..tnakit import statutils
from ..tnakit import typeutils


# Accepted list for response_method
RESPONSE_METHODS = [
    None,
    "nearest",
    "linear",
    "linear-schedule",
    "rbf",
    "rbf-map",
    "rbf-linear",
]
# List of RBF types
RBF_METHODS = [
    "rbf",
    "rbf-map",
    "rbf-linear"
]
# RBF function types
RBF_FUNCS = [
    "multiquadric",
    "inverse_multiquadric",
    "gaussian",
    "linear",
    "cubic",
    "quintic",
    "thin_plate"
]
# Names of parameters needed to describe an RBF network
RBF_SUFFIXES = ["method", "rbf", "func", "eps", "smooth", "N", "xcols"]


# Options for RDBNull
class DataKitOpts(ftypes.BaseFileOpts):
   # --- Global Options ---
    # List of options
    _optlist = {
        "csv",
        "db"
        "mat",
        "simplecsv",
        "simpletsv",
        "textdata",
        "xls"
    }

    # Alternate names
    _optmap = {
        "tsvsimple": "simpletsv",
        "csvsimple": "simplecsv",
        "xlsx": "xlsx",
    }


# Definitions for RDBNull
class DataKitDefn(ftypes.BaseFileDefn):
   # --- Global Options ---
    # Option list
    _optlist = {
        "Dimension",
        "Shape",
    }

    # Alternate names
    _optmap = {
        "dim": "Dimension",
        "ndim": "Dimension",
        "shape": "Shape",
    }

   # --- Types ---
    # Allowed types
    _opttypes = {
        "Dimension": int,
        "Shape": tuple,
    }


# Combine options with parent class
DataKitDefn.combine_optdefs()


# Combine options with parent class
DataKitOpts.combine_optdefs()


# Declare base class
class DataKit(ftypes.BaseData):
    r"""Basic database template without responses

    :Call:
        >>> db = DataKit(fname=None, **kw)
        >>> db = DataKit(db)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            File name; extension is used to guess data format
        *db*: :class:`DataKit`
            DataKit from which to link data and defns
        *csv*: {``None``} | :class:`str`
            Explicit file name for :class:`CSVFile` read
        *textdata*: {``None``} | :class:`str`
            Explicit file name for :class:`TextDataFile`
        *simplecsv*: {``None``} | :class:`str`
            Explicit file name for :class:`CSVSimple`
        *simpletsv*: {``None``} | :class:`str`
            Explicit file name for :class:`TSVSimple`
        *xls*: {``None``} | :class:`str`
            File name for :class:`XLSFile`
        *mat*: {``None``} | :class:`str`
            File name for :class:`MATFile`
    :Outputs:
        *db*: :class:`DataKit`
            Generic database
    :Versions:
        * 2019-12-04 ``@ddalle``: Version 1.0
        * 2020-02-19 ``@ddalle``: Version 1.1; was ``DBResponseNull``
    """
  # =====================
  # Class Attributes
  # =====================
  # <
   # --- Options ---
    # Class for options
    _optscls = DataKitOpts
    # Class for definitions
    _defncls = DataKitDefn

   # --- Special Columns ---
    _tagsubmap = {}

   # --- Response Method Names ---
    # Primary names
    _method_names = [
        "exact",
        "function",
        "multilinear",
        "multilinear-schedule",
        "nearest",
        "rbf",
        "rbf-linear",
        "rbf-map"
    ]

    # Alternates
    _method_map = {
        "fn": "function",
        "func": "function",
        "lin-rbf": "rbf-linear",
        "linear": "multilinear",
        "linear-rbf": "rbf-linear",
        "linear-schedule": "multilinear-schedule",
        "map-rbf": "rbf-map",
        "rbf-global": "rbf",
        "rbf-schedule": "rbf-map",
        "rbf0": "rbf",
        "rbf1": "rbf-map",
    }

    # Method functions
    _method_funcs = {
        0: {
            "exact": "rcall_exact",
            "function": "rcall_function",
            "multilinear": "rcall_multilinear",
            "multilinear-schedule": "rcall_multilinear_schedule",
            "nearest": "rcall_nearest",
            "rbf": "rcall_rbf",
            "rbf-linear": "rcall_rbf_linear",
            "rbf-map": "rcall_rbf_schedule",
        },
        1: {
            "exact": "rcall_exact",
            "function": "rcall_function",
            "multilinear": "rcall_multilinear",
            "multilinear-schedule": "rcall_multilinear_schedule",
            "nearest": "rcall_nearest",
        },
    }

    # Method constructors
    _method_constructors = {
        "function": "_create_function",
        "rbf": "_create_rbf",
        "rbf-linear": "_create_rbf_linear",
        "rbf-map": "_create_rbf_map",
    }
  # >

  # =============
  # Config
  # =============
  # <
   # --- Dunder Methods ---
    # Initialization method
    def __init__(self, fname=None, **kw):
        r"""Initialization method

        :Versions:
            * 2019-12-06 ``@ddalle``: Version 1.0
        """
        # Required attributes
        self.cols = []
        self.n = 0
        self.defns = {}
        self.bkpts = {}
        self.sources = {}
        # Evaluation attributes
        self.response_arg_alternates = {}
        self.response_arg_converters = {}
        self.response_arg_aliases = {}
        self.response_arg_defaults = {}
        self.response_args = {}
        self.response_kwargs = {}
        self.response_methods = {}
        self.response_xargs = {}
        # Radial basis function containers
        self.rbf = {}
        # Extra attributes for plotting
        self.col_pngs = {}
        self.col_seams = {}
        # 1D output image and seam directives
        self.png_fnames = {}
        self.png_figs = {}
        self.png_kwargs = {}
        self.seam_cols = {}
        self.seam_figs = {}
        self.seam_kwargs = {}

        # Process keyword options
        self.opts = self.process_kw(_warnmode=0, **kw)
        # Create a mapped copy for below
        kw = kwutils.map_kw(self._optscls._optmap, **kw)

        # Check for null inputs
        if (fname is None) and (not kw):
            return

        # Check for *db* option
        db = kw.get("db", fname)

        # Get file name extension
        if typeutils.isstr(fname):
            # Get extension
            ext = fname.split(".")[-1]
        elif isinstance(db, DataKit):
            # Link data from another datakit
            self.link_db(db)
            # Stop
            return
        elif fname is not None:
            # Too confusing
            raise TypeError("Non-keyword input must be ``None`` or a string")
        else:
            # No file extension
            ext = None

        # Initialize file name handles for each type
        fcsv  = None
        ftsv  = None
        fcsvs = None
        ftsvs = None
        ftdat = None
        fxls  = None
        fmat  = None
        # Filter *ext*
        if ext == "csv":
            # Guess it's a mid-level CSV file
            fcsv = fname
        elif ext == "tsv":
            # Guess it's a mid-level TSV file
            ftsv = fname
        elif ext == "xls":
            # Guess it's a spreadsheet
            fxls = fname
        elif ext == "xlsx":
            # Guess it's a spreadsheet
            fxls = fname
        elif ext == "mat":
            # Guess it's a MATLAB file
            fmat = fname
        elif ext is not None:
            # Unable to guess
            raise ValueError(
                "Unable to guess file type of file name '%s'" % fname)
        # Last-check file names
        fcsv  = kw.pop("csv", fcsv)
        ftsv  = kw.pop("tsv", ftsv)
        fxls  = kw.pop("xls", fxls)
        fmat  = kw.pop("mat", fmat)
        fcsvs = kw.pop("simplecsv", fcsvs)
        ftsvs = kw.pop("simpletsv", ftsvs)
        ftdat = kw.pop("textdata",  ftdat)

        # Read
        if fcsv is not None:
            # Read CSV file
            self.read_csv(fcsv, **kw)
        elif ftsv is not None:
            # Read TSV file
            self.read_tsv(ftsv, **kw)
        elif fxls is not None:
            # Read XLS file
            self.read_xls(fxls, **kw)
        elif fcsvs is not None:
            # Read simple CSV file
            self.read_csvsimple(fcsvs, **kw)
        elif ftsvs is not None:
            # Read simple TSV file
            self.read_tsvsimple(ftsvs, **kw)
        elif ftdat is not None:
            # Read generic textual data file
            self.read_textdata(ftdat, **kw)
        elif fmat is not None:
            # Read MATLAB file
            self.read_mat(fmat, **kw)
        else:
            # If reaching this point, process values
            self.process_kw_values()

   # --- Copy ---
    # Copy
    def copy(self):
        r"""Make a copy of a database class

        Each database class may need its own version of this class

        :Call:
            >>> dbcopy = db.copy()
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
        :Outputs:
            *dbcopy*: :class:`DataKit`
                Copy of generic database
        :Versions:
            * 2019-12-04 ``@ddalle``: Version 1.0
        """
        # Form a new database
        dbcopy = self.__class__()
        # Copy relevant parts
        self.copy_DataKit(dbcopy)
        # Output
        return dbcopy

    # Link data from another DataKit
    def link_db(self, dbsrc, init=True):
        r"""Link attributes from another DataKit

        :Call:
            >>> qdb = db.link_db(dbsrc, init=True)
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
            *dbsrc*: :class:`DataKit`
                Source database from which to link data
            *init*: {``True``} | ``False``
                Flag if this is used to create initial DataKit
        :Outputs:
            *qdb*: ``True`` | ``False``
                Whether or not *dbsrc* was linked
        :Versions:
            * 2021-07-20 ``@ddalle``: Version 1.0
        """
        # Check *dbsrc* type
        if not isinstance(dbsrc, DataKit):
            # Copy/link unsuccessful
            return False
        # Initialize empty DataKit
        if init:
            DataKit.__init__(self)
        # Loop through cols
        for col in dbsrc.cols:
            # Copy definition first
            self.set_defn(col, dbsrc.get_defn(col))
            # Save values
            self.save_col(col, dbsrc.get_values(col))
        # Copy/link successful
        return True

    # Copy attributes and data known to DataKit class
    def copy_DataKit(self, dbcopy):
        r"""Copy attributes and data relevant to null-response DB

        :Call:
            >>> db.copy_DataKit(dbcopy)
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
            *dbcopy*: :class:`DataKit`
                Copy of generic database
        :Versions:
            * 2019-12-04 ``@ddalle``: Version 1.0
        """
        # Loop through columns
        for col in self.cols:
            dbcopy[col] = copy.copy(self[col])
        # Copy all attributes
        self.copy__dict__(dbcopy, skip=[])

    # Copy any remaining items
    def copy__dict__(self, dbtarg, skip=[]):
        r"""Copy all attributes except for specified list

        :Call:
            >>> db.copy__dict__(dbtarg, skip=[])
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
            *dbtarg*: :class:`DataKit`
                Generic database; target copy
            *skip*: :class:`list`\ [:class:`str`]
                List of attributes not to copy
        :Effects:
            ``getattr(dbtarg, k)``: ``getattr(db, k, vdef)``
                Shallow copy of attribute from *DBc* or *vdef* if necessary
        :Versions:
            * 2019-12-04 ``@ddalle``: Version 1.0
        """
        # Check *skip*
        if not isinstance(skip, (list, tuple)):
            raise TypeError("Attributes to skip during copy must be list")
        # Loop through dict
        for k in self.__dict__:
            # Check if it should be skipped
            if k in skip:
                continue
            # Get copy, if possible
            try:
                # Create the copy
                vcopy = self.copyitem(self.__dict__[k])
                # Set it
                dbtarg.__dict__[k] = vcopy
            except Exception:
                # No copy
                continue

    # Copy an attribute if present
    def copyattr(self, dbtarg, k, vdef={}):
        r"""Make an appropriate copy of an attribute if present

        :Call:
            >>> db.copyattr(dbtarg, k, vdef={})
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
            *dbtarg*: :class:`DataKit`
                Generic database; target copy
            *k*: :class:`str`
                Name of attribute to copy
            *vdef*: {``{}``} | :class:`any`
                Default value for output attribute if ``getattr(db,k)``
                does not exist
        :Effects:
            ``getattr(dbtarg, k)``: ``getattr(db, k, vdef)``
                Shallow copy of attribute from *DBc* or *vdef* if necessary
        :Versions:
            * 2018-06-08 ``@ddalle``: Version 1.0
            * 2019-12-04 ``@ddalle``: Copied from :class:`DBCoeff`
        """
        # Check for attribute
        if hasattr(self, k):
            # Get attribute
            v = getattr(self, k)
        else:
            # Use default
            v = vdef
        # Copy item
        vcopy = self.copyitem(v)
        # Save it
        setattr(dbtarg, k, vcopy)

    # Copy an item according to local rules
    def copyitem(self, v):
        r"""Return a copy of appropriate depth following class rules

        :Call:
            >>> vcopy = db.copyitem(v)
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
            *v*: :class:`any`
                Variable to be copied
        :Outputs:
            *vcopy*: *v.__class__*
                Copy of *v* (shallow or deep)
        :Versions:
            * 2019-12-04 ``@ddalle``: Version 1.0
        """
        # Type
        t = v.__class__
        # Check the type in order to make a copy
        if v is None:
            # Not necessary to copy
            return
        elif t == dict:
            # Deep copy of dictionary
            try:
                return copy.deepcopy(v)
            except Exception:
                return dict(v)
        elif hasattr(v, "copy"):
            # Use class's already-build copy() method
            return v.copy()
        elif t == list:
            # Copy list
            return list(v)
        else:
            # Shallow copy
            return copy.copy(v)
  # >

  # ==================
  # Options
  # ==================
  # <
   # --- Column Definitions ---
    # Set a definition
    def set_defn(self, col, defn, _warnmode=0):
        r"""Set a column definition, with checks

        :Call:
            >>> db.set_defn(col, defn, _warnmode=0)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *col*: :class:`str`
                Data column name
            *defn*: :class:`dict`
                (Partial) definition for *col*
            *_warnmode*: {``0``} | ``1`` | ``2``
                Warning mode for invalid *defn* keys or values
        :Versions:
            * 2020-03-06 ``@ddalle``: Documented
        """
        # Get dictionary of options
        defns = self.__dict__.setdefault("defns", {})
        # Create filtered definition
        defn_checked = self._defncls(_warnmode=_warnmode, **defn)
        # Set definition
        defns[col] = defn_checked

   # --- Copy/Link ---
    # Link options
    def clone_options(self, opts, prefix=""):
        r"""Copy a database's options

        :Call:
            >>> db.clone_options(opts, prefix="")
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *opts*: :class:`dict`
                Options dictionary
            *prefix*: {``""``} | :class:`str`
                Prefix to append to key names in *db.opts*
        :Effects:
            *db.opts*: :class:`dict`
                Options merged with or copied from *opts*
            *db.defns*: :class:`dict`
                Merged with ``opts["Definitions"]``
        :Versions:
            * 2019-12-06 ``@ddalle``: Version 1.0
            * 2019-12-26 ``@ddalle``: Added *db.defns* effect
            * 2020-02-10 ``@ddalle``: Removed *db.defns* effect
            * 2020-03-06 ``@ddalle``: Renamed from :func:`copy_options`
        """
        # Check input
        if not isinstance(opts, dict):
            raise TypeError("Options input must be dict-type")
        # Get options
        dbopts = self.__dict__.setdefault("opts", {})
        # Merge options
        for (k, v) in opts.items():
            # Apply prefix
            if prefix:
                # Add strings; no delimiter
                k1 = prefix + k
            else:
                # No prefix
                k1 = k
            # Check for "Definitions"; handled separately
            if k1 == "Definitions":
                continue
            # Get existing value
            v0 = dbopts.get(k1)
            # Check types
            if isinstance(v, dict) and isinstance(v0, dict):
                # Update dictionary
                v0.update(**v)
            else:
                # Overwrite or add
                dbopts[k] = v

    # Link definitions
    def clone_defns(self, defns, prefix="", _warnmode=0):
        r"""Copy a data store's column definitions

        :Call:
            >>> db.clone_defns(defns, prefix="")
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *defns*: :class:`dict`
                Dictionary of column definitions
            *prefix*: {``""``} | :class:`str`
                Prefix to append to key names in *db.opts*
        :Effects:
            *db.opts*: :class:`dict`
                Options merged with or copied from *opts*
            *db.defns*: :class:`dict`
                Merged with ``opts["Definitions"]``
        :Versions:
            * 2019-12-06 ``@ddalle``: Version 1.0
            * 2019-12-26 ``@ddalle``: Added *db.defns* effect
            * 2020-02-13 ``@ddalle``: Split from :func:`copy_options`
            * 2020-03-06 ``@ddalle``: Renamed from :func:`copy_defns`
        """
        # Check input
        if not isinstance(defns, dict):
            raise TypeError(
                "'defns' input must be 'dict', got '%s'" % defns.__class__)
        # Loop through input definitions
        for (k, defn) in defns.items():
            # Check definition type
            if not isinstance(defn, dict):
                raise TypeError(
                    ("Definition for col '%s' " % k) +
                    ("must be 'dict', got '%s'" % defns.__class__))
            # Apply prefix
            if prefix:
                # Prepend column name
                col = prefix + k
            else:
                # Reuse column name
                col = k
            # Save the definition (in database format)
            self.set_defn(col, defn, _warnmode)

   # --- Definitions: Get ---
    # Get output dimension
    def get_ndim(self, col):
        r"""Get database dimension for column *col*

        :Call:
            >>> ndim = db.get_ndim(col)
        :Inputs:
            *db*: :class:`cape.attdb.rdbscalar.DBResponseLinear`
                Database with multidimensional output functions
            *col*: :class:`str`
                Name of column to evaluate
        :Outputs:
            *ndim*: {``0``} | :class:`int`
                Dimension of *col* in database
        :Versions:
            * 2020-03-12 ``@ddalle``: Version 1.0
        """
        # Get column definition
        defn = self.get_defn(col)
        # Get dimensionality
        ndim = defn.get("Dimension")
        # Check valid result
        if isinstance(ndim, int):
            return ndim
        # Otherwise, get data
        V = self.get_all_values(col)
        # Check
        if isinstance(V, np.ndarray):
            # Get dimensions directly from data
            ndim = V.ndim
            # Save it
            defn["Dimension"] = ndim
            defn["Shape"] = V.shape
        elif V is None:
            # No dimension
            return
        else:
            # List is 1-dimensional
            ndim = 1
        # Output
        return ndim

    # Get output dimension
    def get_output_ndim(self, col):
        r"""Get output dimension for column *col*

        :Call:
            >>> ndim = db.get_output_ndim(col)
        :Inputs:
            *db*: :class:`cape.attdb.rdbscalar.DBResponseLinear`
                Database with multidimensional output functions
            *col*: :class:`str`
                Name of column to evaluate
        :Outputs:
            *ndim*: {``0``} | :class:`int`
                Dimension of *col* at a single condition
        :Versions:
            * 2019-12-27 ``@ddalle``: Version 1.0
            * 2020-03-12 ``@ddalle``: Keyed from "Dimension"
        """
        # Get column dimension
        ndim = self.get_ndim(col)
        # Check for miss
        if ndim is None:
            # No dimension
            return
        else:
            # Subtract one
            return ndim - 1

   # --- Definitions: Set ---
    # Set dimensionality
    def set_ndim(self, col, ndim):
        r"""Set database dimension for column *col*

        :Call:
            >>> db.set_ndim(col, ndim)
        :Inputs:
            *db*: :class:`cape.attdb.rdbscalar.DBResponseLinear`
                Database with multidimensional output functions
            *col*: :class:`str`
                Name of column to evaluate
        :Outputs:
            *ndim*: {``0``} | :class:`int`
                Dimension of *col* in database
        :Versions:
            * 2019-12-30 ``@ddalle``: Version 1.0
        """
        # Get column definition
        defn = self.get_defn(col)
        # Default
        if ndim is None:
            # Assume scalar
            ndim = 1
        # Check type
        if not isinstance(ndim, int):
            raise TypeError(
                "Output dimension for '%s' must be int (got %s)" %
                (col, type(ndim)))
        # Set it
        defn["Dimension"] = ndim

    # Set output dimensionality
    def set_output_ndim(self, col, ndim):
        r"""Set output dimension for column *col*

        :Call:
            >>> db.set_output_ndim(col, ndim)
        :Inputs:
            *db*: :class:`cape.attdb.rdbscalar.DBResponseLinear`
                Database with multidimensional output functions
            *col*: :class:`str`
                Name of column to evaluate
        :Outputs:
            *ndim*: {``0``} | :class:`int`
                Dimension of *col* at a single condition
        :Versions:
            * 2019-12-30 ``@ddalle``: Version 1.0
        """
        # Get column definition
        defn = self.get_defn(col)
        # Default
        if ndim is None:
            # Assume scalar output
            ndim = 0
        # Check type
        if not isinstance(ndim, int):
            raise TypeError(
                "Output dimension for '%s' must be int (got %s)" %
                (col, type(ndim)))
        # Set it
        defn["Dimension"] = ndim + 1
  # >

  # ================
  # Sources
  # ================
  # <
   # --- Get Source ---
    # Get a source by type and number
    def get_source(self, ext=None, n=None):
        r"""Get a source by category (and number), if possible

        :Call:
            >>> dbf = db.get_source(ext)
            >>> dbf = db.get_source(ext, n)
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
            *ext*: {``None``} | :class:`str`
                Source type, by extension, to retrieve
            *n*: {``None``} | :class:`int` >= 0
                Source number
        :Outputs:
            *dbf*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
        :Versions:
            * 2020-02-13 ``@ddalle``: Version 1.0
        """
        # Get sources
        srcs = self.__dict__.get("sources", {})
        # Check for *n*
        if n is None:
            # Check for both ``None``
            if ext is None:
                raise ValueError("Either 'ext' or 'n' must be specified")
            # Loop through sources
            for name, dbf in srcs.items():
                # Check name
                if name.split("-")[1] == ext:
                    # Output
                    return dbf
            else:
                # No match
                return
        elif ext is None:
            # Check names
            targ = "%02i" % n
            # Loop through sources
            for name, dbf in srcs.items():
                # Check name
                if name.split("-")[0] == targ:
                    # Output
                    return dbf
            else:
                # No match
                return
        else:
            # Get explicit name
            name = "%02i-%s" % (n, ext)
            # Check for source
            return srcs.get(name)

    # Get source, creating if necessary
    def make_source(self, ext, cls, n=None, cols=None, save=True, **kw):
        r"""Get or create a source by category (and number)

        :Call:
            >>> dbf = db.make_source(ext, cls)
            >>> dbf = db.make_source(ext, cls, n=None, cols=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
            *ext*: :class:`str`
                Source type, by extension, to retrieve
            *cls*: :class:`type`
                Subclass of :class:`BaseFile` to create (if needed)
            *n*: {``None``} | :class:`int` >= 0
                Source number to search for
            *cols*: {*db.cols*} | :class:`list`\ [:class:`str`]
                List of data columns to include in *dbf*
            *save*: {``True``} | ``False``
                Option to save *dbf* in *db.sources*
            *attrs*: {``None``} | :class:`list`\ [:class:`str`]
                Extra attributes of *db* to save for ``.mat`` files
        :Outputs:
            *dbf*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
        :Versions:
            * 2020-02-13 ``@ddalle``: Version 1.0
            * 2020-03-06 ``@ddalle``: Rename from :func:`get_dbf`
        """
        # Don't use existing if *cols* is specified
        if cols is None and kw.get("attrs") is None:
            # Get the source
            dbf = self.get_source(ext, n=n)
            # Check if found
            if dbf is not None:
                # Done
                return dbf
        # Create a new one
        dbf = self.genr8_source(ext, cls, cols=cols, **kw)
        # Save the file interface if needed
        if save:
            # Name for this source
            name = "%02i-%s" % (len(self.sources), ext)
            # Save it
            self.sources[name] = dbf
        # Output
        return dbf

    # Build new source, creating if necessary
    def genr8_source(self, ext, cls, cols=None, **kw):
        r"""Create a new source file interface

        :Call:
            >>> dbf = db.genr8_source(ext, cls)
            >>> dbf = db.genr8_source(ext, cls, cols=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
            *ext*: :class:`str`
                Source type, by extension, to retrieve
            *cls*: :class:`type`
                Subclass of :class:`BaseFile` to create (if needed)
            *cols*: {*db.cols*} | :class:`list`\ [:class:`str`]
                List of data columns to include in *dbf*
            *attrs*: {``None``} | :class:`list`\ [:class:`str`]
                Extra attributes of *db* to save for ``.mat`` files
        :Outputs:
            *dbf*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
        :Versions:
            * 2020-03-06 ``@ddalle``: Split from :func:`make_source`
        """
        # Default columns
        if cols is None:
            # Use listed columns
            cols = list(self.cols)
        # Get relevant options
        kwcls = {"_warnmode": 0}
        # Set values
        kwcls["Values"] = {col: self[col] for col in cols}
        # Explicit column list
        kwcls["cols"] = cols
        # Copy definitions
        kwcls["Definitions"] = self.defns
        # Create from class
        dbf = cls(**kwcls)
        # Get attributes to copy
        attrs = kw.get("attrs")
        # Copy them
        self._copy_attrs(dbf, attrs)
        # Output
        return dbf

    # Copy attributes
    def _copy_attrs(self, dbf, attrs):
        r"""Copy additional attributes to new "source" database

        :Call:
            >>> db._copy_attrs(dbf, attrs)
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
            *dbf*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                Data file interface
            *attrs*: ``None`` | :class:`list`\ [:class:`str`]
                List of *db* attributes to copy
        :Versions:
            * 2020-04-30 ``@ddalle``: Version 1.0
        """
        # Check for null option
        if attrs is None:
            return
        # Loop through attributes
        for attr in attrs:
            # Get current value
            v = self.__dict__.get(attr)
            # Check for :class:`dict`
            if not isinstance(v, dict):
                # Copy attribute and move to next attribute
                setattr(dbf, attr, copy.copy(v))
                continue
            # Check if this is a dict of information by column
            if not any([col in v for col in dbf]):
                # Just some other :class:`dict`; copy whole hting
                setattr(dbf, attr, copy.copy(v))
            # Initialize dict to save
            v1 = {}
            # Loop through cols
            for col, vi in v.items():
                # Check if it's a *col* in *dbf*
                if col in dbf:
                    v1[col] = copy.copy(vi)
            # Save new :class:`dict`
            setattr(dbf, attr, v1)
  # >

  # ==================
  # I/O
  # ==================
  # <
   # --- CSV ---
    # Read CSV file
    def read_csv(self, fname, **kw):
        r"""Read data from a CSV file

        :Call:
            >>> db.read_csv(fname, **kw)
            >>> db.read_csv(dbcsv, **kw)
            >>> db.read_csv(f, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
            *fname*: :class:`str`
                Name of CSV file to read
            *dbcsv*: :class:`cape.attdb.ftypes.csvfile.CSVFile`
                Existing CSV file
            *f*: :class:`file`
                Open CSV file interface
            *append*: ``True`` | {``False``}
                Option to combine cols with same name
            *save*, *SaveCSV*: ``True`` | {``False``}
                Option to save the CSV interface to *db._csv*
        :See Also:
            * :class:`cape.attdb.ftypes.csvfile.CSVFile`
        :Versions:
            * 2019-12-06 ``@ddalle``: Version 1.0
        """
        # Get option to save database
        save = kw.pop("save", kw.pop("SaveCSV", False))
        # Set warning mode
        kw.setdefault("_warnmode", 0)
        # Check input type
        if isinstance(fname, ftypes.CSVFile):
            # Already a CSV database
            dbf = fname
        else:
            # Create an instance
            dbf = ftypes.CSVFile(fname, **kw)
        # Link the data
        self.link_data(dbf, append=kw.get("append", False))
        # Copy the options
        self.clone_defns(dbf.defns)
        # Apply default
        self.finish_defns(dbf.cols)
        # Save the file interface if needed
        if save:
            # Name for this source
            name = "%02i-csv" % len(self.sources)
            # Save it
            self.sources[name] = dbf

    # Write dense CSV file
    def write_csv_dense(self, fname, cols=None):
        r""""Write dense CSV file

        If *db.sources* has a CSV file, the database will be written
        from that object.  Otherwise, :func:`make_source` is called.

        :Call:
            >>> db.write_csv_dense(fname, cols=None)
            >>> db.write_csv_dense(f, cols=None)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *fname*: :class:`str`
                Name of file to write
            *f*: :class:`file`
                File open for writing
            *cols*: {*db.cols*} | :class:`list`\ [:class:`str`]
                List of columns to write
        :Versions:
            * 2019-12-06 ``@ddalle``: Version 1.0
            * 2020-02-14 ``@ddalle``: Uniform "sources" interface
        """
        # Get CSV file interface
        dbcsv = self.make_source("csv", ftypes.CSVFile, cols=cols)
        # Write it
        dbcsv.write_csv_dense(fname, cols=cols)

    # Write (nice) CSV file
    def write_csv(self, fname, cols=None, **kw):
        r""""Write CSV file with full options

        If *db.sources* has a CSV file, the database will be written
        from that object.  Otherwise, :func:`make_source` is called.

        :Call:
            >>> db.write_csv(fname, cols=None, **kw)
            >>> db.write_csv(f, cols=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *fname*: :class:`str`
                Name of file to write
            *f*: :class:`file`
                File open for writing
            *cols*: {*db.cols*} | :class:`list`\ [:class:`str`]
                List of columns to write
            *kw*: :class:`dict`
                Keyword args to :func:`CSVFile.write_csv`
        :Versions:
            * 2020-04-01 ``@ddalle``: Version 1.0
        """
        # Get CSV file interface
        dbcsv = self.make_source("csv", ftypes.CSVFile, cols=cols)
        # Write it
        dbcsv.write_csv(fname, cols=cols, **kw)

   # --- Simple CSV ---
    # Read simple CSV file
    def read_csvsimple(self, fname, **kw):
        r"""Read data from a simple CSV file

        :Call:
            >>> db.read_csvsimple(fname, **kw)
            >>> db.read_csvsimple(dbcsv, **kw)
            >>> db.read_csvsimple(f, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
            *fname*: :class:`str`
                Name of CSV file to read
            *dbcsv*: :class:`cape.attdb.ftypes.csvfile.CSVSimple`
                Existing CSV file
            *f*: :class:`file`
                Open CSV file interface
            *save*, *SaveCSV*: ``True`` | {``False``}
                Option to save the CSV interface to *db._csv*
        :See Also:
            * :class:`cape.attdb.ftypes.csvfile.CSVFile`
        :Versions:
            * 2019-12-06 ``@ddalle``: Version 1.0
        """
        # Get option to save database
        savecsv = kw.pop("save", kw.pop("SaveCSV", False))
        # Set warning mode
        kw.setdefault("_warnmode", 0)
        # Check input type
        if isinstance(fname, ftypes.CSVSimple):
            # Already a CSV database
            dbf = fname
        else:
            # Create an instance
            dbf = ftypes.CSVSimple(fname, **kw)
        # Link the data
        self.link_data(dbf)
        # Copy the definitions
        self.clone_defns(dbf.defns)
        # Apply default
        self.finish_defns(dbf.cols)
        # Save the file interface if needed
        if save:
            # Name for this source
            name = "%02i-csvsimple" % len(self.sources)
            # Save it
            self.sources[name] = dbf

   # --- TSV ---
    # Read TSV file
    def read_tsv(self, fname, **kw):
        r"""Read data from a space-separated file

        :Call:
            >>> db.read_tsv(fname, **kw)
            >>> db.read_tsv(dbtsv, **kw)
            >>> db.read_tsv(f, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
            *fname*: :class:`str`
                Name of TSV file to read
            *dbcsv*: :class:`cape.attdb.ftypes.tsvfile.TSVFile`
                Existing TSV file
            *f*: :class:`file`
                Open TSV file handle
            *append*: ``True`` | {``False``}
                Option to combine cols with same name
            *save*, *SaveTSV*: ``True`` | {``False``}
                Option to save the TSV interface to *db.sources*
        :See Also:
            * :class:`cape.attdb.ftypes.tsvfile.CSVFile`
        :Versions:
            * 2019-12-06 ``@ddalle``: Version 1.0 (:func:`read_csv`)
            * 2021-01-14 ``@ddalle``: Version 1.0
        """
        # Get option to save database
        save = kw.pop("save", kw.pop("SaveTSV", False))
        # Set warning mode
        kw.setdefault("_warnmode", 0)
        # Check input type
        if isinstance(fname, ftypes.TSVFile):
            # Already a CSV database
            dbf = fname
        else:
            # Create an instance
            dbf = ftypes.TSVFile(fname, **kw)
        # Link the data
        self.link_data(dbf, append=kw.get("append", False))
        # Copy the options
        self.clone_defns(dbf.defns)
        # Apply default
        self.finish_defns(dbf.cols)
        # Save the file interface if needed
        if save:
            # Name for this source
            name = "%02i-tsv" % len(self.sources)
            # Save it
            self.sources[name] = dbf

    # Write dense TSV file
    def write_tsv_dense(self, fname, cols=None):
        r""""Write dense TSV file

        If *db.sources* has a TSV file, the database will be written
        from that object.  Otherwise, :func:`make_source` is called.

        :Call:
            >>> db.write_tsv_dense(fname, cols=None)
            >>> db.write_tsv_dense(f, cols=None)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *fname*: :class:`str`
                Name of file to write
            *f*: :class:`file`
                File open for writing
            *cols*: {*db.cols*} | :class:`list`\ [:class:`str`]
                List of columns to write
        :Versions:
            * 2019-12-06 ``@ddalle``: Version 1.0 (write_csv_dense)
            * 2021-01-14 ``@ddalle``: Version 1.0
        """
        # Get TSV file interface
        dbtsv = self.make_source("tsv", ftypes.TSVFile, cols=cols)
        # Write it
        dbtsv.write_tsv_dense(fname, cols=cols)

    # Write (nice) TSV file
    def write_tsv(self, fname, cols=None, **kw):
        r""""Write TSV file with full options

        If *db.sources* has a TSV file, the database will be written
        from that object.  Otherwise, :func:`make_source` is called.

        :Call:
            >>> db.write_tsv(fname, cols=None, **kw)
            >>> db.write_tsv(f, cols=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *fname*: :class:`str`
                Name of file to write
            *f*: :class:`file`
                File open for writing
            *cols*: {*db.cols*} | :class:`list`\ [:class:`str`]
                List of columns to write
            *kw*: :class:`dict`
                Keyword args to :func:`TSVFile.write_tsv`
        :Versions:
            * 2020-04-01 ``@ddalle``: Version 1.0 (write_csv)
            * 2021-01-14 ``@ddalle``: Version 1.0
        """
        # Get TSV file interface
        dbtsv = self.make_source("tsv", ftypes.TSVFile, cols=cols)
        # Write it
        dbtsv.write_tsv(fname, cols=cols, **kw)

   # --- Simple TSV ---
    # Read simple TSV file
    def read_tsvsimple(self, fname, **kw):
        r"""Read data from a simple TSV file

        :Call:
            >>> db.read_tsvsimple(fname, **kw)
            >>> db.read_tsvsimple(dbcsv, **kw)
            >>> db.read_tsvsimple(f, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
            *fname*: :class:`str`
                Name of TSV file to read
            *dbtsv*: :class:`cape.attdb.ftypes.tsvfile.TSVSimple`
                Existing TSV file
            *f*: :class:`file`
                Open TSV file interface
            *save*, *SaveTSV*: ``True`` | {``False``}
                Option to save the TSV interface to *db.sources*
        :See Also:
            * :class:`cape.attdb.ftypes.tsvfile.TSVFile`
        :Versions:
            * 2019-12-06 ``@ddalle``: Version 1.0 (read_csvsimple)
            * 2021-01-14 ``@ddalle``: Version 1.0
        """
        # Get option to save database
        save = kw.pop("save", kw.pop("SaveTSV", False))
        # Set warning mode
        kw.setdefault("_warnmode", 0)
        # Check input type
        if isinstance(fname, ftypes.TSVSimple):
            # Already a CSV database
            dbf = fname
        else:
            # Create an instance
            dbf = ftypes.TSVSimple(fname, **kw)
        # Link the data
        self.link_data(dbf)
        # Copy the definitions
        self.clone_defns(dbf.defns)
        # Apply default
        self.finish_defns(dbf.cols)
        # Save the file interface if needed
        if save:
            # Name for this source
            name = "%02i-tsvsimple" % len(self.sources)
            # Save it
            self.sources[name] = dbf

   # --- Text Data ---
    # Read text data fiel
    def read_textdata(self, fname, **kw):
        r"""Read data from a simple CSV file

        :Call:
            >>> db.read_textdata(fname, **kw)
            >>> db.read_textdata(dbcsv, **kw)
            >>> db.read_textdata(f, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
            *fname*: :class:`str`
                Name of CSV file to read
            *dbcsv*: :class:`cape.attdb.ftypes.textdata.TextDataFile`
                Existing CSV file
            *f*: :class:`file`
                Open CSV file interface
            *save*: {``True``} | ``False``
                Option to save the CSV interface to *db._csv*
        :See Also:
            * :class:`cape.attdb.ftypes.csvfile.CSVFile`
        :Versions:
            * 2019-12-06 ``@ddalle``: Version 1.0
        """
        # Get option to save database
        savedat = kw.pop("save", False)
        # Set warning mode
        kw.setdefault("_warnmode", 0)
        # Check input type
        if isinstance(fname, ftypes.TextDataFile):
            # Already a file itnerface
            dbf = fname
        else:
            # Create an insteance
            dbf = ftypes.TextDataFile(fname, **kw)
        # Linke the data
        self.link_data(dbf)
        # Copy the definitions
        self.clone_defns(dbf.defns)
        # Apply default
        self.finish_defns(dbf.cols)
        # Save the file interface if needed
        if savedat:
            # Name for this source
            name = "%02i-textdata" % len(self.sources)
            # Save it
            self.sources[name] = dbf

   # --- XLS ---
    # Read XLS file
    def read_xls(self, fname, **kw):
        r"""Read data from an ``.xls`` or ``.xlsx`` file

        :Call:
            >>> db.read_xls(fname, **kw)
            >>> db.read_xls(dbxls, **kw)
            >>> db.read_xls(wb, **kw)
            >>> db.read_xls(ws, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
            *dbxls*: :class:`cape.attdb.ftypes.xls.XLSFile`
                Existing XLS file interface
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
            *save*, *SaveXLS*: ``True`` | {``False``}
                Option to save the XLS interface to *db._xls*
        :See Also:
            * :class:`cape.attdb.ftypes.xls.XLSFile`
        :Versions:
            * 2019-12-06 ``@ddalle``: Version 1.0
        """
        # Get option to save database
        save = kw.pop("save", kw.pop("SaveXLS", False))
        # Set warning mode
        kw.setdefault("_warnmode", 0)
        # Check input type
        if isinstance(fname, ftypes.XLSFile):
            # Already a CSV database
            dbf = fname
        else:
            # Create an instance
            dbf = ftypes.XLSFile(fname, **kw)
        # Link the data
        self.link_data(dbf)
        # Copy the definitions
        self.clone_defns(dbf.defns)
        # Apply default
        self.finish_defns(dbf.cols)
        # Save the file interface if needed
        if save:
            # Name for this source
            name = "%02i-xls" % len(self.sources)
            # Save it
            self.sources[name] = dbf

    # Write XLSX file
    def write_xls(self, fname, cols=None, **kw):
        r""""Write XLS file with full options

        If *db.sources* has a XLS file, the database will be written
        from that object.  Otherwise, :func:`make_source` is called.

        :Call:
            >>> db.write_xls(fname, cols=None, **kw)
            >>> db.write_xls(wb, cols=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *fname*: :class:`str`
                Name of file to write
            *wb*: :class:`xlsxwriter.Workbook`
                Opened XLS workbook
            *cols*: {*db.cols*} | :class:`list`\ [:class:`str`]
                List of columns to write
            *kw*: :class:`dict`
                Keyword args to :func:`CSVFile.write_csv`
        :Versions:
            * 2020-05-21 ``@ddalle``: Version 1.0
        """
        # Get column names per sheet
        sheetcols = kw.get("sheetcols", {})
        # Full set of such cols
        if cols is None:
            # Default: all columns
            cols = self.cols
        # Initialize with specified cols
        allcols = list(cols)
        # Combine all columns from *sheetcols*
        for _cols in sheetcols.values():
            allcols.update(_cols)
        # Get XLS file interface
        dbxls = self.make_source("xls", ftypes.XLSFile, cols=allcols)
        # Write it
        dbxls.write_xls(fname, cols=cols, **kw)

   # --- MAT ---
    # Read MAT file
    def read_mat(self, fname, **kw):
        r"""Read data from a version 5 ``.mat`` file

        :Call:
            >>> db.read_mat(fname, **kw)
            >>> db.read_mat(dbmat, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
            *fname*: :class:`str`
                Name of ``.mat`` file to read
            *dbmat*: :class:`cape.attdb.ftypes.mat.MATFile`
                Existing MAT file interface
            *save*, *SaveMAT*: ``True`` | {``False``}
                Option to save the MAT interface to *db._mat*
        :See Also:
            * :class:`cape.attdb.ftypes.mat.MATFile`
        :Versions:
            * 2019-12-17 ``@ddalle``: Version 1.0
        """
        # Get option to save database
        save = kw.pop("save", kw.pop("SaveMAT", False))
        # Set warning mode
        kw.setdefault("_warnmode", 0)
        # Check input type
        if isinstance(fname, ftypes.MATFile):
            # Already a MAT database
            dbf = fname
        else:
            # Create an instance
            dbf = ftypes.MATFile(fname, **kw)
        # Columns to keep
        cols = []
        # Make replacements for column names
        for (j, col) in enumerate(dbf.cols):
            # Check name
            if col.startswith("DB."):
                # Strip prefix from name
                col1 = col[3:]
                # Replace key
                dbf[col1] = dbf.pop(col)
                # Save this column
                cols.append(col1)
            elif col.startswith("bkpts."):
                # Strip "bkpts" from name
                col1 = col[6:]
                # Create break points
                bkpts = self.__dict__.setdefault("bkpts", {})
                # Save them
                bkpts[col1] = dbf.pop(col)
            else:
                # No change; save this column
                cols.append(col)
        # Subset column list
        dbf.cols = cols
        # Link the data
        self.link_data(dbf)
        # Copy the definitions
        self.clone_defns(dbf.defns)
        # Apply default
        self.finish_defns(dbf.cols)
        # Link other attributes
        for (k, v) in dbf.__dict__.items():
            # Check if present and nonempty
            if self.__dict__.get(k):
                continue
            # Otherwise link
            self.__dict__[k] = v
        # Save the file interface if needed
        if save:
            # Name for this source
            name = "%02i-mat" % len(self.sources)
            # Save it
            self.sources[name] = dbf

    # Write MAT file
    def write_mat(self, fname, cols=None, **kw):
        r""""Write a MAT file

        If *db.sources* has a MAT file, the database will be written
        from that object.  Otherwise, :func:`make_source` is called.

        :Call:
            >>> db.write_mat(fname, cols=None)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *fname*: :class:`str`
                Name of file to write
            *f*: :class:`file`
                File open for writing
            *cols*: {*db.cols*} | :class:`list`\ [:class:`str`]
                List of columns to write
        :Versions:
            * 2019-12-06 ``@ddalle``: Version 1.0
        """
        # Attributes
        attrs = kw.get("attrs", ["bkpts"])
        # Get/create MAT file interface
        dbmat = self.make_source("mat", ftypes.MATFile, cols=cols, attrs=attrs)
        # Write it
        dbmat.write_mat(fname, cols=cols, attrs=attrs)

   # --- RBF specials ---
    def infer_rbfs(self, cols, **kw):
        r"""Infer radial basis function responses for several *cols*

        :Call:
            >>> db.infer_rbfs(cols, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                DataKit where *db.rbf[col]* will be defined
            *cols*: :class:`list`\ [:class:`str`]
                Name of column whose RBF will be constructed
            *xcols*: {``None``} | :class:`list`\ [:class:`str`]
                Explicit list of arguments for all *cols*
        :See Also:
            * :func:`infer_rbf`
            * :func:`create_rbf_cols`
        :Versions:
            * 2021-09-16 ``@ddalle``: Version 1.0
        """
        # Infer RBF for each *col*
        for col in cols:
            self.infer_rbf(col, **kw)

    # Infer RBF from cols with expected suffixes
    def infer_rbf(self, col, vals=None, **kw):
        r"""Infer a radial basis function response mechanism

        This looks for columns with specific suffixes in order to create
        a Radial Basis Function (RBF) response mechanism in *db*.
        Suppose that *col* is ``"CY"`` for this example, then this
        function will look for the following columns, either in *col* or
        *vals*:

            * ``"CY"``: nominal values at which RBF was created
            * ``"CY_method"``: response method index
            * ``"CY_rbf"``: weights of RBF nodes
            * ``"CY_func"``: RBF basis function index
            * ``"CY_eps"``: scaling parameter for (each) RBF
            * ``"CY_smooth:``: RBF smoothing parameter
            * ``"CY_N"``: number of nodes in (each) RBF
            * ``"CY_xcols"``: explicit list of RBF argument names
            * ``"CY_X"``: 2D matrix of values of RBF args
            * ``"CY_x0"``: values of first argument if not global RBF

        The *CY_method* column will repeat one of the following values:

            * ``4``: ``"rbf"``
            * ``5``: ``"rbf-map"``
            * ``6``: ``"rbf-schedule"``

        The *CY_func* legend is as follows:

            * ``0``: ``"multiquadric"``
            * ``1``: ``"inverse_multiquadric"``
            * ``2``: ``"gaussian"``
            * ``3``: ``"linear"``
            * ``4``: ``"cubic"``
            * ``5``: ``"quintic"``
            * ``6``: ``"thin_plate"``
        
        :Call:
            >>> db.infer_rbf(col, vals=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                DataKit where *db.rbf[col]* will be defined
            *col*: :class:`str`
                Name of column whose RBF will be constructed
            *vals*: :class:`dict`\ [:class:`np.ndarray`]
                Data to use in RBF creation in favor of *db*
        :Effects:
            *db[col]*: :class:`np.ndarray`\ [:class:`float`]
                Values of *col* used in RBF
            *db[xcol]*: :class:`np.ndarray`\ [:class:`float`]
                Values of RBF args saved for each *xcol*
            *db.bkpts[xcol]*: :class:`np.ndarray`\ [:class:`float`]
                Break points for each RBF arg
            *db.rbf[col]*: :class:`Rbf` | :class:`list`
                One or more SciPy radial basis function instances
            *db.response_methods[col]*: :class:`str`
                Name of inferred response method
        :Versions:
            * 2021-09-16 ``@ddalle``: Version 1.0
        """
       # --- Options ---
       # --- Get values ---
        # Default *vals* dict of extra columns
        if vals is None:
            vals = {}
        # Special column names
        col0 = col
        col2 = "%s_rbf" % col
        col3 = "%s_func" % col
        col4 = "%s_eps" % col
        col5 = "%s_smooth" % col
        col6 = "%s_N" % col
        # Extract values
        v = vals.get(col0, self.get(col0))
        v_rbf  = vals.get(col2, self.get(col2))
        v_func = vals.get(col3, self.get(col3))
        v_eps  = vals.get(col4, self.get(col4))
        v_smth = vals.get(col5, self.get(col5))
        v_N = vals.get(col6, self.get(col6))
        # Get arg values
        v_X = self._infer_response_x(col, vals, **kw)
        # Get list of arguments
        xcols = self._infer_xcols(col, vals, **kw)
        # Get response method
        imeth, eval_meth = self._infer_response_method(col, vals)
        # Check if method is an RBF
        if eval_meth not in RBF_METHODS:
            raise ValueError(
                "eval_method '%s' (index %i) is not an RBF"
                % (eval_meth, imeth))
       # --- RBFs ---
        # (Total) number of test points
        nx = v.size
        # Entries for uniform cols
        nx2 = v_eps.size
        # Number of RBF args
        narg = v_X.shape[1]
        # Create dummy inputs for blank RBF
        Z = ([0.0, 1.0],) * narg
        # Number of RBFs (potentially)
        if eval_meth == "rbf":
            # %%% Global RBF %%%
            # Read function type
            func = RBF_FUNCS[int(v_func[0])]
            # Create RBF
            rbf = scirbf.Rbf(*Z, function=func)
            # Save not-too-important actual values
            rbf.di = v
            # Use all conditions
            rbf.xi = v_X.T
            # Get RBF weights
            rbf.nodes = v_rbf
            # Get function type
            rbf.function = func
            # Get scale factor
            rbf.epsilon = v_eps[0]
            # Smoothing parameter
            rbf.smooth = v_smth[0]
            # Save count
            rbf.N = nx
            # Save RBF
            self.rbf[col] = rbf
            # Save args to main db (overwrite)
            for j, xcol in enumerate(xcols):
                # Get values
                xj = v_X[:, j]
                # Save them
                self.save_col(xcol, xj)
            # Create break point tables
            self.create_bkpts(xcols, nmin=1)
        else:
            # %%% Multiple RBFS %%%
            # Get *x0* values
            v_x0 = self._infer_response_x0(col, vals, **kw)
            # Get unique values of first *xcol*
            x0_bkpts = np.unique(v_x0)
            # Count unique values
            nrbf = x0_bkpts.size
            # Expand *x0* if needed
            if nx != nx2:
                # "expand" *v_x0* by repeating each value *v_N* times
                v_x0 = np.hstack(
                    [np.full(v_N[j], x0_bkpts[j]) for j in range(nrbf)])
            # Initialize db.rbf
            self.rbf[col] = []
            # Loop through RBF slices
            for j, x0j in enumerate(x0_bkpts):
                # Indices for this beginning of slice
                if j == 0:
                    ia = 0
                else:
                    ia = ib
                # Index from repeated cols
                if nx == nx2:
                    # Parameters like *eps* repeated *rbf.N* times
                    j2 = ia
                else:
                    # Parameters like *eps* only reported once per RBF
                    j2 = j
                # End of slice
                ib = ia + int(v_N[j2])
                # Read function type
                func = RBF_FUNCS[int(v_func[j2])]
                # Create RBF
                rbf = scirbf.Rbf(*Z, function=func)
                # Save not-too-important actual values
                rbf.di = v[ia:ib]
                # Use all conditions
                rbf.xi = v_X[ia:ib,:].T
                # Get RBF weights
                rbf.nodes = v_rbf[ia:ib]
                # Get function type
                rbf.function = func
                # Get scale factor
                rbf.epsilon = v_eps[j2]
                # Smoothing parameter
                rbf.smooth = v_smth[j2]
                # Save count
                rbf.N = v_N[j2]
                # Save RBF
                self.rbf[col].append(rbf)
            # Save first arg
            self.save_col(xcols[0], v_x0)
            # Save args to main db (overwrite)
            for j, xcol in enumerate(xcols[1:]):
                # Get values
                xj = v_X[:, j]
                # Save them
                self.save_col(xcol, xj)
            # Get break points
            self.create_bkpts(xcols[0], nmin=1)
            self.create_bkpts_schedule(xcols[1:], xcols[0], nmin=1)
       # --- Final Definitions ---
        # Save main values
        self.save_col(col, v)
        # Save evaluation method
        self.set_response_method(col, eval_meth)
        # Save response mechanism arguments
        self.set_response_args(col, xcols)

    # Generate special cols for an RBF
    def create_rbfs_cols(self, cols, **kw):
        r"""Save data to describe multiple existing RBFs

        :Call:
            >>> db.create_rbfs_cols(cols, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                DataKit with *db.rbf[col]* defined
            *cols*: :class:`str`
                Name of columns whose RBFs will be archived
            *expand*: ``True`` | {``False``}
                Repeat properties like *eps* for each node of RBF
                (for uniform data size, usually to write to CSV file)
        :See Also:
            * :func:`create_rbf_cols`
            * :func:`infer_rbfs`
            * :func:`infer_rbf`
        :Versions:
            * 2021-09-16 ``@ddalle``: Version 1.0
        """
        # Loop through column list
        for col in cols:
            self.create_rbf_cols(col, **kw)

    def create_rbf_cols(self, col, **kw):
        r"""Generate data to describe existing RBF(s) for *col*

        This saves various properties extracted from *db.rbf[col]*
        directly as additional columns in *db*. These values can then be
        used but :func:`infer_rbf` to reconstruct a SciPy radial basis
        function response mechanism without re-solving the original
        linear system of equations that trains the RBF weights.
        
        :Call:
            >>> db.create_rbf_cols(col, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                DataKit with *db.rbf[col]* defined
            *col*: :class:`str`
                Name of column whose RBF will be analyzed
            *expand*: ``True`` | {``False``}
                Repeat properties like *eps* for each node of RBF
                (for uniform data size, usually to write to CSV file)
        :Effects:
            *db[col]*: :class:`np.ndarray`\ [:class:`float`]
                Values of *col* used in RBF
            *db[col+"_method"]*: :class:`np.ndarray`\ [:class:`int`]
                Response method index:

                * ``4``: ``"rbf"``
                * ``5``: ``"rbf-map"``
                * ``6``: ``"rbf-schedule"``

            *db[col+"_rbf"]*: :class:`np.ndarray`\ [:class:`float`]
                Weight for each RBF node
            *db[col+"_func"]*: :class:`np.ndarray`\ [:class:`int`]
                RBF basis function index:

                * ``0``: ``"multiquadric"``
                * ``1``: ``"inverse_multiquadric"``
                * ``2``: ``"gaussian"``
                * ``3``: ``"linear"``
                * ``4``: ``"cubic"``
                * ``5``: ``"quintic"``
                * ``6``: ``"thin_plate"``

            *db[col+"_eps"]*: :class:`np.ndarray`\ [:class:`float`]
                Epsilon scaling factor for (each) RBF
            *db[col+"_smooth"]*: :class:`np.ndarray`\ [:class:`float`]
                Smoothing factor for (each) RBF
            *db[col+"_N"]*: :class:`np.ndarray`\ [:class:`float`]
                Number of nodes in (each) RBF
            *db[col+"_xcols"]*: :class:`list`\ [:class:`str`]
                List of arguments for *col*
            *db.response_args[col]*: :class:`list`\ [:class:`str`]
                List of arguments for *col*
        :Versions:
            * 2021-09-16 ``@ddalle``: Version 1.0
        """
        # Create results
        vals = self.genr8_rbf_cols(col, **kw)
        # Check if nominal values saved
        if col not in self:
            # Get values from *vals*
            v = vals[col]
            # Save
            self.save_col(col, v)
        # Column index to keep *col* descriptors close
        index_col = self.cols.index(col) + 1
        # Suffixes to link from *vals*
        for suffix in reversed(RBF_SUFFIXES):
            # Special col name
            colj = "%s_%s" % (col, suffix)
            # Append this column
            if colj not in self.cols:
                self.cols.insert(index_col, colj)
            # Save the values from *vals* dict
            self.save_col(colj, vals[colj])
        
    def genr8_rbf_cols(self, col, **kw):
        r"""Generate data to describe existing RBF(s) for *col*

        This creates a :class:`dict` of various properties that are used
        by the radial basis function (or list thereof) within *db.rbf*.
        It is possible to recreate an RBF(s) with only this information,
        thus avoiding the need to retrain the RBF network(s).
        
        :Call:
            >>> vals = db.genr8_rbf_cols(col, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                DataKit with *db.rbf[col]* defined
            *col*: :class:`str`
                Name of column whose RBF will be analyzed
            *expand*: ``True`` | {``False``}
                Repeat properties like *eps* for each node of RBF
                (for uniform data size, usually to write to CSV file)
        :Outputs:
            *vals*: :class:`dict`\ [:class:`np.ndarray`]
                Data used in *db.rbf[col]*
            *vals[col]*: :class:`np.ndarray`\ [:class:`float`]
                Values of *col* used in RBF (may differ from *db[col]*)
            *vals[col+"_method"]*: :class:`np.ndarray`\ [:class:`int`]
                Response method index:

                * ``4``: ``"rbf"``
                * ``5``: ``"rbf-map"``
                * ``6``: ``"rbf-schedule"``

            *vals[col+"_rbf"]*: :class:`np.ndarray`\ [:class:`float`]
                Weight for each RBF node
            *vals[col+"_func"]*: :class:`np.ndarray`\ [:class:`int`]
                RBF basis function index:

                * ``0``: ``"multiquadric"``
                * ``1``: ``"inverse_multiquadric"``
                * ``2``: ``"gaussian"``
                * ``3``: ``"linear"``
                * ``4``: ``"cubic"``
                * ``5``: ``"quintic"``
                * ``6``: ``"thin_plate"``

            *vals[col+"_eps"]*: :class:`np.ndarray`\ [:class:`float`]
                Epsilon scaling factor for (each) RBF
            *vals[col+"_smooth"]*: :class:`np.ndarray`\ [:class:`float`]
                Smoothing factor for (each) RBF
            *vals[col+"_N"]*: :class:`np.ndarray`\ [:class:`float`]
                Number of nodes in (each) RBF
            *vals[col+"_x0"]*: :class:`np.ndarray`\ [:class:`float`]
                Values of first response arg if *db.rbf[col]* is a list
            *vals[col+"_X"]*: :class:`np.ndarray`\ [:class:`float`]
                2D matrix of node locations for (each) RBF
            *vals[col+"_x.<xcol>"]*: :class:`np.ndarray`
                1D array of node location values for each response arg
            *vals[col+"_xcols"]*: :class:`list`\ [:class:`str`]
                List of arguments for *col*
        :Versions:
            * 2021-09-15 ``@ddalle``: Version 1.0
        """
       # --- Options ---
        # Expand option
        expand = kw.get("expand", False)
       # --- Checks ---
        # Get arg list for first *coeff*
        eval_args = self.get_response_args(col)
        # Check for validity
        if eval_args is None:
            # No eval args for this coeff
            raise KeyError("No 'response_args' for col '%s'" % col)
        # Check if present
        if col not in self:
            raise KeyError("No coeff '%s' in database" % k)
        # Get evaluation type
        eval_meth = self.get_response_method(col)
        # Check it
        if eval_meth is None:
            raise KeyError("No 'response_method' for col '%s'" % col)
        # Check for RBFs attribute
        try:
            self.rbf[col]
        except AttributeError:
            # No definitions at all
            raise AttributeError("No RBFs defined for this datakit")
        except KeyError:
            # No RBFs saved for ref coeff
            raise KeyError("No RBFs for col '%s'" % col)
        # Check if the method is in our accepted list
        if eval_meth not in RBF_METHODS:
            raise ValueError(
                ("Response method '%s' is for col " % eval_meth) +
                ("'%s' is not an RBF method" % col))
        # Number of arguments
        narg = len(eval_args)
        # First arg
        xcol0 = eval_args[0]
       # --- Init ---
        # Initialize values
        vals = {}
        cols = []
        # Integer code for method
        imeth = RESPONSE_METHODS.index(eval_meth)
        # Check type
        if eval_meth == "rbf":
            # Get single global rbf
            rbf = self.rbf[col]
            rbfs = [rbf]
        else:
            # Get list of rbfs
            rbfs = self.rbf[col]
        # Number of rbfs
        nrbf = len(rbfs)
        # Sizes
        nxs = np.array([rbfj.nodes.size for rbfj in rbfs])
        ixs = np.cumsum(nxs)
        nx = np.sum(nxs)
        # Special column names
        col0 = col
        col1 = "%s_method" % col
        col2 = "%s_rbf" % col
        col3 = "%s_func" % col
        col4 = "%s_eps" % col
        col5 = "%s_smooth" % col
        col6 = "%s_N" % col
        col7 = "%s_X" % col
        col8 = "%s_x0" % col
        col9 = "%s_xcols" % col
        # Initialize values
        if expand:
            # Expand all cols to same size
            nx2 = nx
        else:
            # Just repeat function index, epsilon, etc. once per RBF
            nx2 = nrbf
        # Initialize values in *vals*
        vals[col0] = np.zeros(nx)
        vals[col1] = np.zeros(nx2, dtype="int32")
        vals[col2] = np.zeros(nx)
        vals[col3] = np.zeros(nx2, dtype="int32")
        vals[col4] = np.zeros(nx2)
        vals[col5] = np.zeros(nx2)
        vals[col6] = np.zeros(nx2, dtype="int64")
        # Initialize arg values
        if eval_meth == "rbf":
            # All variables treated in like manner
            vals[col7] = np.zeros((nx, narg))
        else:
            # Fist *xcol* is special (fixed for each RBF)
            vals[col7] = np.zeros((nx, narg-1))
            vals[col8] = np.zeros(nx2)
       # --- Data ---
        # Loop through RBFs
        for i, rbf in enumerate(rbfs):
            # Indices for this slice
            ia = ixs[i] - nxs[i]
            ib = ixs[i]
            # Save collapsible fields
            if expand:
                # Save same value for every row of *col0*
                ii = slice(ia, ib)
            else:
                # Save single value for reach RBF
                ii = slice(i, i+1)
            # Save main values
            vals[col0][ia:ib] = rbf.di
            # Save RBF weights
            vals[col2][ia:ib] = rbf.nodes
            # Save node locations (arg values)
            vals[col7][ia:ib,:] = rbf.xi.T
            # Check for map- or schedule-rbf slice col
            if eval_meth == "rbf":
                pass
            else:
                # Save the slice value
                vals[col8][ii] = self.bkpts[xcol0][i]
            # All x-values pa
            # Save method index
            vals[col1][ii] = imeth
            # Save RBF basis function index
            vals[col3][ii] = RBF_FUNCS.index(rbf.function)
            # Save epsilon scaling factor
            vals[col4][ii] = rbf.epsilon
            # Save smoothing factor
            vals[col5][ii] = rbf.smooth
            # Save sice
            vals[col6][ii] = rbf.N
        # Save additional entries for each *xcol*
        for j, xcol in enumerate(eval_args):
            # Special column name
            xcolj = "%s_x.%s" % (col, xcol)
            # Check RBF type
            if eval_meth == "rbf":
                # Uniform behavior for all *xcols*
                vals[xcolj] = vals[col7][:, j]
            else:
                # Handle first column separately
                if j == 0:
                    # Link "x0" col
                    vals[xcolj] = vals[col8]
                else:
                    # Save values from "nodes"
                    vals[xcolj] = vals[col7][:, j-1]
        # Save arguments
        vals[col9] = eval_args
        # Output
        return vals

    # Convert data object to RBF
    def create_rbf_from_db(self, dbf):
        r"""Create RBF response from data object

        :Call:
            >>> db.create_rbf_from_db(dbf)
        :Inputs:
            *db*: :class:`DataKit`
                Data container with responses
            *dbf*: :class:`dict` | :class:`BaseData`
                Raw data container
        :Versions:
            * 2019-07-24 ``@ddalle``: Version 1.0; :func:`ReadRBFCSV`
            * 2021-06-07 ``@ddalle``: Version 2.0
            * 2021-09-14 ``@ddalle``: Version 2.1; bug fix/testing
        """
       # --- Checks ---
        # Test that it's a **dict**
        if not isinstance(dbf, dict):
            raise TypeError(
                "Data object is not a dict or subclass, got '%s'" % type(dbf))
        # Test for column named "eval_method"
        if "eval_method" not in dbf:
            raise KeyError("Data object has not column named 'eval_method'")
        # Get list of columns
        try:
            # Use attribute from DataKit (preserves order)
            cols = dbf.cols
        except AttributeError:
            # Copy the keys (this will not work in Python 2.x)
            cols = list(dbf.keys())
        # Number of args determined by # of cols before *eval_method*
        narg = cols.index("eval_method")
        # Get that column
        imeth = dbf["eval_method"]
        # Check for valid method
        if np.any(imeth != imeth[0]):
            raise ValueError("Column 'eval_method' must have uniform values")
        # Get first entry
        imeth = int(imeth[0])
        # Check validity
        if (imeth < 0) or (imeth >= len(RBF_METHODS)):
            raise ValueError("Invalid 'eval_method' %s" % imeth)
       # --- Args ---
        # Store first columns as evaluation args
        arg_cols = cols[:narg]
        # Store names of output coefficients
        coeff_cols = cols[narg+1::5]
        # Initialize prefixed/suffixed names
        args = []
        coeffs = []
       # --- Save values ---
        # Loop through arg columns
        for (j, col) in enumerate(arg_cols + coeff_cols):
            # True column number
            if j < narg:
                # First *narg* columns are lookup variables
                i = j
            else:
                # After *eval_method*,
                i = narg + 1 + (j-narg)*5
            # Save name
            if j < narg:
                # Save independent variable name
                args.append(col)
            else:
                # Save dependent variable name
                coeffs.append(col)
            # Check for duplication
            if col in self.cols:
                raise ValueError(
                    "RBF output col '%s' is already present." % coeff)
            # Save the data
            self.save_col(col, dbf[col])
       # --- Create RBFs ---
        # Dereference evaluation method list
        eval_meth = RBF_METHODS[imeth]
        # Get values of first arg
        X0 = dbf[args[0]] 
        # Get number of test points
        nx = X0.size
        # Create matrix of input conditions
        X = np.zeros((nx, narg))
        # Get slice locations if appropriate
        if eval_meth == "rbf":
            # Create dummy inputs for blank RBF
            Z = ([0.0, 1.0],) * narg
        else:
            # Create dummy inputs for blank RBF
            Z = ([0.0, 1.0],) * (narg - 1)
            # Get unique values of first *arg*
            xs = np.unique(dbf[args[0]])
        # Loop through coefficients
        for j, coeff in enumerate(coeffs):
            # Save evaluation arguments
            self.set_response_args(coeff, args)
            # Save evaluation method
            self.set_response_method(coeff, eval_meth)
            # Check for slices
            if eval_meth == "rbf":
                # %%% Global RBF %%%
                # Read function type
                func = RBF_FUNCS[int(dbf[col + "_func"][0])]
                # Create RBF
                rbf = scirbf.Rbf(*Z, function=func)
                # Save not-too-important actual values
                rbf.di = dbf[coeff]
                # Use all conditions
                rbf.xi = X.T
                # Get RBF weights
                rbf.nodes = dbf[coeff + "_rbf"]
                # Get function type
                rbf.function = func
                # Get scale factor
                rbf.epsilon = dbf[coeff + "_eps"][0]
                # Smoothing parameter
                rbf.smooth = dbf[coeff + "_smooth"][0]
                # Save count
                rbf.N = nx
                # Save RBF
                self.rbf[coeff] = rbf
            else:
                # Initialize RBF list
                self.rbf[coeff] = []
                # Cumulative row index
                i0 = 0
                # Slice RBFs
                for i, xi in enumerate(xs):
                    # Get function type
                    func = RBF_FUNCS[int(dbf[col + "_func"][0])]
                    # Initialize slice RBF
                    rbf = scirbf.Rbf(*Z, function=func)
                    # Number of points in this slice
                    ni = np.count_nonzero(X0 == xi)
                    # End of range
                    i1 = i0 + ni
                    # Use subset of conditions
                    rbf.xi = X[i0:i1, :].T
                    # Save not-too-important original values
                    rbf.di = dbf[coeff][i0:i1]
                    # Get RBF weights
                    rbf.nodes = dbf[coeff + "_rbf"][i0:i1]
                    # Get function type
                    rbf.function = func
                    # Get scale factor
                    rbf.epsilon = dbf[coeff + "_eps"][i0]
                    # Smoothing parameter
                    rbf.smooth = dbf[coeff + "_smooth"][i0]
                    # Save count
                    rbf.N = ni
                    # Save RBF
                    self.rbf[coeff].append(rbf)
                    # Update indices
                    i0 = i1
        # Check for global RBFs or schedule
        if eval_meth == "rbf":
            # Break points for each independent variable
            self.create_bkpts(args)
        else:
            # Break points for mapping variable
            self.create_bkpts(args[0:1])
            # Get break points at each slice
            self.create_bkpts_schedule(args[1:], args[0])

    # Read RBF directly from CSV file
    def read_rbf_csv(self, fname, **kw):
        r"""Read RBF directly from a CSV file
        
        :Call:
            >>> db.read_rbf_csv(fname, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Generic database
            *fname*: :class:`str`
                Name of CSV file to read
        :See Also:
            * :class:`cape.attdb.ftypes.csvfile.CSVFile`
        :Versions:
            * 2021-06-17 ``@ddalle``: Version 1.0
        """
        # Set warning mode
        kw.setdefault("_warnmode", 0)
        # Read the data
        dbf = ftypes.CSVFile(fname, **kw)
        # Create RBF from data
        self.create_rbf_from_db(dbf)

    # Write a CSV file for radial basis functions
    def write_rbf_csv(self, fcsv, coeffs, **kw):
        r"""Write an ASCII file of radial basis func coefficients

        :Call:
            >>> db.WriteRBFCSV(fcsv, coeffs=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data container with responses
            *fcsv*: :class:`str`
                Name of ASCII data file to write
            *coeffs*: :class:`list`\ [:class:`str`]
                List of output coefficients to write
            *fmts*: :class:`dict` | :class:`str`
                Dictionary of formats to use for each *coeff*
            *comments*: {``"#"``} | :class:`str`
                Comment character, used as first character of file
            *delim*: {``", "``} | :class:`str`
                Delimiter
            *translators*: {``{}``} | :class:`dict`
                Dictionary of coefficient translations, e.g.
                *CAF* -> *CA*
        :Versions:
            * 2019-07-24 ``@ddalle``: Version 1.0; :func:`WriteRBFCSV`
            * 2021-06-09 ``@ddalle``: Version 2.0
        """
       # --- Checks ---
        # Check input type
        if not isinstance(coeffs, (list, tuple)):
            raise TypeError("Coefficient list must be a list of strings")
        elif len(coeffs) < 1:
            raise ValueError("Coefficient list must have at least one entry")
        # Reference coefficient
        k = coeffs[0]
        # Check if present
        if k not in self:
            raise KeyError("No coeff '%s' in database" % k)
        # Get length
        n = len(self[k])
        # Get arg list for first *coeff*
        eval_args = self.get_response_args(k)
        # Check for validity
        if eval_args is None:
            # No eval args for this coeff
            raise KeyError("No 'response_args' for col '%s'" % k)
        # Get evaluation type
        eval_meth = self.get_response_method(k)
        # Check it
        if eval_meth is None:
            raise KeyError("No 'response_method' for col '%s'" % k)
        # Check for RBFs attribute
        try:
            self.rbf[k]
        except AttributeError:
            # No definitions at all
            raise AttributeError("No RBFs defined for this database")
        except KeyError:
            # No RBFs saved for ref coeff
            raise KeyError("No RBFs for coeff '%s'" % k)
        # Check if the method is in our accepted list
        if eval_meth not in RBF_METHODS:
            raise ValueError(
                "Eval method '%s' is not consistent with RBFs" % eval_meth)
        # Loop through the keys
        for k in coeffs[1:]:
            # Check presence of required information
            if k not in self.cols:
                raise KeyError("No col '%s' in database" % k)
            elif self.get_response_args(k) is None:
                raise KeyError("No 'response_args' for col '%s'" % k)
            elif self.get_response_method(k) is None:
                raise KeyError("No 'response_method' for col '%s'" % k)
            elif k not in self.rbf:
                raise KeyError("No RBFs for coeff '%s'" % k)
            # Check length
            if len(self[k]) != n:
                raise ValueError(
                    "Col '%s' has length %i; expected %i"
                    % (k, len(self[k]), n))
            # Check values
            if self.get_response_args(k) != eval_args:
                raise ValueError("Mismatching response_args for col '%s'" % k)
            elif self.get_response_method(k) != eval_meth:
                raise ValueError(
                    ("Mismatching response_method '%s' for col '%s'" % (
                        self.get_response_args(k), k)) +
                    ("; expected '%s'" % eval_meth))
       # --- Init ---
        # Number of arguments
        narg = len(eval_args)
        # Get method index
        imeth = RBF_METHODS.index(eval_meth)
        # Initialize final column list
        cols = eval_args + ["eval_method"]
        # Initialize values
        vals = {}
        # Extra columns for each output coeff
        COEFF_COLS = ["", "rbf", "func", "eps", "smooth"]
        # Loop through coefficients
        for k in coeffs:
            # Loop through suffixes to initialize columns
            for suf in COEFF_COLS:
                # Append suffix
                col = ("%s_%s" % (k, suf)).rstrip("_")
                # Add actual value at that point
                cols.append(col)
                # Get RBF
                if imeth == 0:
                    # Get global rbf
                    rbf = self.rbf[k]
                    # No RBF list
                    nrbf = 1
                    # Fixed size
                    nx = rbf.nodes.size
                else:
                    # Get first mapped/linear rbf slice+
                    rbfs = self.rbf[k]
                    rbf = self.rbf[k][0]
                    # Number of RBF slices
                    nrbf = len(rbfs)
                    # Total number of nodes
                    nx = np.sum([rbfj.nodes.size for rbfj in rbfs])
                # Initialize data
                vals[col] = np.zeros(nx)
        # Loop through independent variables
        for col in eval_args:
            # Initialize the values
            vals[col] = np.zeros(nx)
        # Append type
        for col in ["eval_method"]:
            # Create values
            vals[col] = np.full(nx, imeth) 
       # --- Data/Properties ---
        # Number of columns
        ncol = len(cols)
        # Total number of points so fart
        ny = 0
        # Loop through RBF slices (same for each *coeff*)
        for j in range(nrbf):
            # Reference coefficient
            coeff = coeffs[0]
            # Get RBF handle
            if imeth == 0:
                # Global RBF
                rbf = self.rbf[coeff]
            else:
                # Slice RBF
                rbf = self.rbf[coeff][j]
            # Number of points in slice
            nj = rbf.nodes.size
            # Save the argument values
            for i, col in enumerate(eval_args):
                if imeth == 0:
                    # Save all the position variables
                    vals[col] = rbf.xi.T[:,i]
                else:
                    # Save the slice points
                    if i == 0:
                        # Scheduling parameter
                        vals[col][ny:ny+nj] = self.bkpts[eval_args[0]][j]
                    else:
                        # Secondary arg
                        vals[col][ny:ny+nj] = rbf.xi.T[:,i-1]
            # Loop through coefficients
            for i, coeff in enumerate(coeffs):
                # Get the RBF
                if imeth == 0:
                    # Global RBF
                    rbf = self.rbf[coeff]
                else:
                    # Slice RBF
                    rbf = self.rbf[coeff][j]
                # Derived columns
                ccols = [
                    ("%s_%s" % (coeff, suf)).rstrip("_") for suf in COEFF_COLS
                ]
                # Save values
                vals[ccols[0]][ny:ny+nj] = rbf.di
                # Save weights
                vals[ccols[1]][ny:ny+nj] = rbf.nodes
                # RBF type
                vals[ccols[2]][ny:ny+nj] = RBF_FUNCS.index(rbf.function)
                # Save spacing parameter
                vals[ccols[3]][ny:ny+nj] = rbf.epsilon
                # Save smoothing parameter
                vals[ccols[4]][ny:ny+nj] = rbf.smooth
            # Update counter
            ny += nj
        # Create CSV file
        dbcsv = ftypes.CSVFile(vals=vals)
        # Ensure proper order
        dbcsv.cols = cols
        # Write it
        dbcsv.write_csv(fcsv, **kw)

    # Determine *xcols* (including order) from existing data
    def _infer_xcols(self, col, vals=None, **kw):
        r"""Infer list of arguments to implied response mech for *col*

        :Call:
            >>> xcols = db._infer_xcols(col, vals=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data interface
            *col*: :class:`str`
                Name of *col* whose response mechanism is queried
            *vals*: {``None``} | :class:`dict`
                Additional values, e.g. from :func:`genr8_rbf_cols`
            *xcols*: {``None``} | :class:`list`\ [:class:`str`]
                Prespecified argument list
        :Outputs:
            *xcols*: {``None``} | :class:`list`\ [:class:`str`]
                Argument list for *col* implied by which cols are
                present in *db* and *vals*
        :Versions:
            * 2021-09-16 ``@ddalle``: Version 1.0
        """
        # Default *vals*
        if vals is None:
            vals = {}
        # Check for specified *xcols* directly in *kw*
        xcols = kw.get("xcols")
        # Check if valid
        if xcols is not None:
            return xcols
        # Special column name
        colx = "%s_xcols" % col
        # Check for this column in several places
        xcols = kw.get(colx, vals.get(colx, self.get(colx)))
        # Use it if appropriate
        if xcols is not None:
            return [xcol.strip() for xcol in xcols]
        # Number of args
        narg = self._infer_response_narg(col, vals)
        # Special column name pattern, like "CYR_x.dy"
        regex = re.compile("%s_x.(\w+)" % col)
        # Loop through current cols to see if we can find args
        xcols = []
        for xcol in self.cols:
            # Check for match
            match = regex.fullmatch(xcol)
            if match is None:
                continue
            # Append column name
            xcols.append(match.group(1))
        # Check if we got the right amount
        if len(xcols) == narg:
            return xcols
        # Look for predefined args
        xcols = self.get_response_args(col)
        # Check for match
        if xcols is not None and len(xcols) == narg:
            return xcols
        # Last chance ... loop through *vals*
        xcols = []
        for xcol in vals:
            # Check for match
            match = regex.fullmatch(xcol)
            if match is None:
                continue
            # Append column name
            xcols.append(match.group(1))
        # Check if we got the right amount
        if len(xcols) == narg:
            return xcols
        # Make dummy arg names
        xcols = ["%s_x.col%i" % (col, i) for i in range(narg)]
        return xcols

    def _infer_response_narg(self, col, vals=None, **kw):
        r"""Infer number of response mechanism arguments

        :Call:
            >>> narg = db._infer_response_narg(col, vals)
        :Inputs:
            *db*: :class:`DataKit`
                Data interface
            *col*: :class:`str`
                Name of *col* whose response mechanism is queried
            *vals*: {``None``} | :class:`dict`
                Additional values, e.g. from :func:`genr8_rbf_cols`
        :Outputs:
            *narg*: :class:`int`
                Number of args to *col* response mechanism
        :Versions:
            * 2021-09-16 ``@ddalle``: Version 1.0
        """
        # Default *vals*
        if vals is None:
            vals = {}
        # Special column name
        colx = "%s_xcols" % col
        # Check for this column in several places
        xcols = kw.get(colx, vals.get(colx, self.get(colx)))
        # Use it if appropriate
        if xcols is not None:
            return len(xcols)
        # Get *X* to see how many args are expected
        colX = "%s_X" % col
        X = vals.get(colX, self.get(colX))
        # Check validity
        if not isinstance(X, np.ndarray):
            raise TypeError(
                "Found no '%s' col to determine number of args" % colX)
        elif X.ndim != 2:
            raise ValueError(
                "Expected 2D '%s' col; got %i dims" % (colX, X.ndim))
        # Get response method
        _, eval_meth = self._infer_response_method(col, vals)
        # Number of args
        if eval_meth == "rbf":
            # Global RBF
            narg = X.shape[1]
        else:
            # Multiple RBFs
            narg = X.shape[1] + 1
        # Output
        return narg

    def _infer_response_x(self, col, vals=None, **kw):
        r"""Infer argument values to response method

        :Call:
            >>> X = db._infer_response_x(col, vals)
        :Inputs:
            *db*: :class:`DataKit`
                Data interface
            *col*: :class:`str`
                Name of *col* whose response mechanism is queried
            *vals*: {``None``} | :class:`dict`
                Additional values, e.g. from :func:`genr8_rbf_cols`
        :Outputs:
            *X*: :class:`np.ndarray`\ [:class:`float`]
                2D array of arg values for all non-slice args
        :Versions:
            * 2021-09-16 ``@ddalle``: Version 1.0
        """
        # Default *vals*
        if vals is None:
            vals = {}
        # Get *X* to see how many args are expected
        colX = "%s_X" % col
        X = vals.get(colX, self.get(colX))
        # Check validity
        if isinstance(X, np.ndarray):
            # Check dimension
            if X.ndim != 2:
                raise ValueError(
                    "Expected 2D '%s' col; got %i dims" % (colX, X.ndim))
            # Valid saved values
            return X
        # Look for explicit kwark
        xcols = kw.get("xcols")
        # Check if that worked
        if xcols is None:
            # Special column name
            colx = "%s_xcols" % col
            # Check for this column in several places
            xcols = kw.get(colx, vals.get(colx, self.get(colx)))
        # Use it if appropriate
        if xcols is None:
            raise ValueError("Unable to infer arg values for '%s'" % col)
        # Ensure list
        xcols = [xcol.strip() for xcol in xcols]
        # Get response method
        _, eval_meth = self._infer_response_method(col, vals)
        # Number of args
        if eval_meth != "rbf":
            # First key is a mapped parameter
            xcols.pop(0)
        # Number of args
        narg = len(xcols)
        # Get size of first arg
        x0 = vals.get(xcols[0], self.get(xcols[0]))
        # Ensure validity
        if x0 is None:
            raise ValueError(
                "Unable to infer arg '%s' values for '%s'" % (xcols[0], col))
        # Get number of test points
        nx = x0.size
        # Initialize output
        X = np.zeros((nx, narg))
        # Save values
        for j, xcol in enumerate(xcols):
            X[:, j] = vals.get(xcol, self.get(xcol))
        # Output
        return X

    def _infer_response_x0(self, col, vals=None, **kw):
        r"""Infer first argument's values to response method

        :Call:
            >>> x0 = db._infer_response_x(col, vals)
        :Inputs:
            *db*: :class:`DataKit`
                Data interface
            *col*: :class:`str`
                Name of *col* whose response mechanism is queried
            *vals*: {``None``} | :class:`dict`
                Additional values, e.g. from :func:`genr8_rbf_cols`
        :Outputs:
            *x0*: :class:`np.ndarray`\ [:class:`float`]
                1D array of arg values for all first (slice) arg
        :Versions:
            * 2021-09-16 ``@ddalle``: Version 1.0
        """
        # Default *vals*
        if vals is None:
            vals = {}
        # Get *X* to see how many args are expected
        colx = "%s_x0" % col
        x0 = vals.get(colx, self.get(colx))
        # Check validity
        if isinstance(x0, np.ndarray):
            # Valid saved values
            return x0
        # Look for explicit kwark
        xcols = kw.get("xcols")
        # Check if that worked
        if xcols is None:
            # Special column name
            colx = "%s_xcols" % col
            # Check for this column in several places
            xcols = kw.get(colx, vals.get(colx, self.get(colx)))
        # Use it if appropriate
        if xcols is None:
            raise ValueError("Unable to infer arg values for '%s'" % col)
        # Get name of first column
        xcol0 = xcols[0].strip()
        # Get size of first arg
        x0 = vals.get(xcol0, self.get(xcol0))
        # Ensure validity
        if x0 is None:
            raise ValueError(
                "Unable to infer arg '%s' values for '%s'" % (xcol0, col))
        # Output
        return x0

    def _infer_response_method(self, col, vals=None):
        r"""Determine response method based on expected *col* names

        :Call:
            >>> i, method = db._infer_response_method(col, vals)
        :Inputs:
            *db*: :class:`DataKit`
                Data interface
            *col*: :class:`str`
                Name of *col* whose response mechanism is queried
            *vals*: {``None``} | :class:`dict`
                Additional values, e.g. from :func:`genr8_rbf_cols`
        :Outputs:
            *i*: :class:`int`
                Index of response method
            *method*: :class:`str`
                Name of response method
        :Versions:
            * 2021-09-16 ``@ddalle``: Version 1.0
        """
        # Default *vals*
        if vals is None:
            vals = {}
        # Column name
        col1 = "%s_method" % col
        # Extract values
        v_method = vals.get(col1, self.get(col1))
        # Check validity
        if not isinstance(v_method, np.ndarray):
            raise TypeError("Missing evaluation method col '%s'" % col1)
        # Get method
        i_method = v_method[0]
        # Expand code
        try:
            response_method = RESPONSE_METHODS[int(i_method)]
        except Exception:
            raise ValueError("Invalid response_method code '%s'" % i_method)
        # Output
        return i_method, response_method
  # >

  # ==================
  # Eval/Call
  # ==================
  # <
   # --- Evaluation ---
    # Evaluate interpolation
    def __call__(self, *a, **kw):
        r"""Generic evaluation function

        :Call:
            >>> v = db(*a, **kw)
            >>> v = db(col, x0, x1, ...)
            >>> V = db(col, x0, X1, ...)
            >>> v = db(col, k0=x0, k1=x1, ...)
            >>> V = db(col, k0=x0, k1=X1, ...)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *x0*: :class:`float` | :class:`int`
                Numeric value for first argument to *col* response
            *x1*: :class:`float` | :class:`int`
                Numeric value for second argument to *col* response
            *X1*: :class:`np.ndarray`\ [:class:`float`]
                Array of *x1* values
            *k0*: :class:`str` | :class:`unicode`
                Name of first argument to *col* response
            *k1*: :class:`str` | :class:`unicode`
                Name of second argument to *col* response
        :Outputs:
            *v*: :class:`float` | :class:`int`
                Function output for scalar evaluation
            *V*: :class:`np.ndarray`\ [:class:`float`]
                Array of function outputs
        :Versions:
            * 2019-01-07 ``@ddalle``: Version 1.0
            * 2019-12-30 ``@ddalle``: Version 2.0: map of methods
        """
       # --- Argument Types ---
        # Process coefficient name and remaining coeffs
        col, a, kw = self._prep_args_colname(*a, **kw)
        # Determine call mode
        mode = self._check_callmode(col, *a, **kw)
        # Filter mode
        if mode == 0:
            # Return entire column
            return self.get_all_values(col)
        elif mode == 1:
            # Use a mask
            return self.get_values(col, a[0])
        elif mode == 2:
            # Use defined response
            return self.rcall(col, *a, **kw)
        elif mode == 3:
            # Use exact; args defined but no method
            args = self.get_response_args(col)
            # Get exact matches
            return self.rcall_exact(col, args, *a, **kw)

    # Evaluate response
    def rcall(self, *a, **kw):
        r"""Evaluate predefined response method

        :Call:
            >>> v = db.rcall(*a, **kw)
            >>> v = db.rcall(col, x0, x1, ...)
            >>> V = db.rcall(col, x0, X1, ...)
            >>> v = db.rcall(col, k0=x0, k1=x1, ...)
            >>> V = db.rcall(col, k0=x0, k1=X1, ...)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *x0*: :class:`float` | :class:`int`
                Numeric value for first argument to *col* response
            *x1*: :class:`float` | :class:`int`
                Numeric value for second argument to *col* response
            *X1*: :class:`np.ndarray`\ [:class:`float`]
                Array of *x1* values
            *k0*: :class:`str` | :class:`unicode`
                Name of first argument to *col* response
            *k1*: :class:`str` | :class:`unicode`
                Name of second argument to *col* response
        :Outputs:
            *v*: :class:`float` | :class:`int`
                Function output for scalar evaluation
            *V*: :class:`np.ndarray`\ [:class:`float`]
                Array of function outputs
        :Versions:
            * 2019-01-07 ``@ddalle``: Version 1.0
            * 2019-12-30 ``@ddalle``: Version 2.0: map of methods
            * 2020-04-20 ``@ddalle``: Moved meat from :func:`__call__`
        """
       # --- Get coefficient name ---
        # Process coefficient
        col, a, kw = self._prep_args_colname(*a, **kw)
       # --- Get method and other parameters ---
        # Specific method
        method_col = self.get_response_method(col)
        # Specific lookup arguments (and copy it)
        args_col = self.get_response_args(col)
        # Get extra args passed along to evaluator
        kw_fn = self.get_response_kwargs(col)
        # Attempt to get default aliases
        arg_aliases = self.get_response_arg_aliases(col)
       # --- Aliases ---
        # Process aliases in *kw*
        for k in dict(kw):
            # Check if there's an alias for *k*
            if k not in arg_aliases:
                continue
            # Get alias for keyword *k*
            alias_k = arg_aliases[k]
            # Save the value under the alias as well
            kw[alias_k] = kw.pop(k)
       # --- Argument values ---
        # Initialize lookup point
        x = []
        # Loop through arguments
        for i, k in enumerate(args_col):
            # Get value
            xi = self.get_arg_value(i, k, *a, **kw)
            # Save it
            x.append(np.asarray(xi))
        # Normalize arguments
        X, dims = self.normalize_args(x)
        # Maximum dimension
        nd = len(dims)
        # Data size
        nx = np.prod(dims)
       # --- Evaluation ---
        # Get class handle
        cls = self.__class__
        # Use lower case with hyphens instead of underscores
        method_col = method_col.lower().replace("_", "-")
        # Get proper method name (default to same)
        method_col = cls._method_map.get(method_col, method_col)
        # Args
        method_args = self.get_response_args(col)
        # Output dimensionality
        ndim_col = self.get_output_ndim(col)
        # Get maps from function name to function callable
        method_funcs = cls._method_funcs[ndim_col]
        # Check if present
        if method_col not in method_funcs:
            # Get close matches
            mtchs = difflib.get_close_matches(
                method_col, list(method_funcs.keys()))
            # Error message
            raise ValueError(
                ("No %i-D eval method '%s'; " % (ndim_col, method_col)) +
                ("closest matches: %s" % mtches))
        # Get the function handle
        f = getattr(self, method_funcs.get(method_col))
        # Combine args (should there be an attribute for this?)
        kw_fn = dict(kw_fn, **kw)
        # Calls
        if nd == 0:
            # Scalar call
            v = f(col, args_col, *x, **kw_fn)
            # Output
            return v
        else:
            # Initialize output
            V = np.zeros(nx)
            # Loop through points
            for j in range(nx):
                # Construct inputs
                xj = [Xi[j] for Xi in X]
                # Construct kwargs
                kwj = dict()
                # Loop through keywords
                for kj, vj in kw_fn.items():
                    # Check if it's a recognized arg
                    if kj in method_args:
                        # Check type
                        if isinstance(vj, (list, np.ndarray)):
                            # Save item
                            kwj[kj] = vj[j]
                        else:
                            # Save whole
                            kwj[kj] = vj
                    else:
                        # Save whole, even if array
                        kwj[kj] = vj
                # Call scalar (hopefully) function
                Vj = f(col, args_col, *xj, **kwj)
                # Check if scalar
                if isinstance(Vj, np.ndarray):
                    # Something allowed an array arg in; try item
                    try:
                        V[j] = Vj[j]
                    except IndexError:
                        V[j] = float(Vj)
                else:
                    # It should be a scalar, so this is the normal sit
                    V[j] = Vj
            # Reshape
            V = V.reshape(dims)
            # Output
            return V

   # --- Alternative Evaluation ---
    # Find exact match
    def rcall_exact(self, col, args, *a, **kw):
        r"""Evaluate a coefficient by looking up exact matches

        :Call:
            >>> v = db.rcall_exact(col, args, *a, **kw)
            >>> V = db.rcall_exact(col, args, *a, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of explanatory col names (numeric)
            *a*: :class:`tuple`\ [:class:`float` | :class:`np.ndarray`]
                Tuple of values for each argument in *args*
            *tol*: {``1.0e-4``} | :class:`float` > 0
                Default tolerance for exact match
            *tols*: {``{}``} | :class:`dict`\ [:class:`float` > 0]
                Dictionary of key-specific tolerances
            *kw*: :class:`dict`\ [:class:`float` | :class:`np.ndarray`]
                Alternate keyword arguments
        :Outputs:
            *v*: ``None`` | :class:`float`
                Value of *db[col]* exactly matching conditions *a*
            *V*: :class:`np.ndarray`\ [:class:`float`]
                Multiple values matching exactly
        :Versions:
            * 2018-12-30 ``@ddalle``: Version 1.0
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
            * 2020-04-24 ``@ddalle``: Switched args to :class:`tuple`
            * 2020-05-19 ``@ddalle``: Support for 2D cols
        """
        # Check for column
        if (col not in self.cols) or (col not in self):
            # Missing col
            raise KeyError("Col '%s' is not present" % col)
        # Get values
        V = self[col]
        # Array length
        if isinstance(V, np.ndarray):
            # Use the last dimension
            n = V.shape[-1]
            # Get dimensionality
            ndim = V.ndim
        else:
            # Use a a simple length
            n = len(V)
            # Always 1D
            ndim = 1
        # Create mask
        I = np.arange(len(V))
        # Tolerance dictionary
        tols = kw.get("tols", {})
        # Default tolerance
        tol = 1.0e-4
        # Loop through keys
        for (i, k) in enumerate(args):
            # Get value
            xi = a[i]
            # Get tolerance
            toli = tols.get(k, kw.get("tol", tol))
            # Apply test
            qi = np.abs(self[k][I] - xi) <= toli
            # Combine constraints
            I = I[qi]
            # Break if no matches
            if len(I) == 0:
                return None
        # Test number of outputs
        if len(I) == 1:
            # Single output
            if ndim == 1:
                # Scalar value
                return V[I[0]]
            elif ndim == 2:
                # Single column
                return V[:, I[0]]
        else:
            # Multiple outputs
            if ndim == 1:
                # Array of scalar values
                if isinstance(V, np.ndarray):
                    # Actual array
                    return V[I]
                else:
                    # Reconstruct list
                    return [V[i] for i in I]
            elif ndim == 2:
                # Several columns
                return V[:, I]

    # Lookup nearest value
    def rcall_nearest(self, col, args, *a, **kw):
        r"""Evaluate a coefficient by looking up nearest match

        :Call:
            >>> v = db.rcall_nearest(col, args, *a, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of (numeric) column to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of explanatory col names (numeric)
            *a*: :class:`tuple`\ [:class:`float` | :class:`np.ndarray`]
                Tuple of values for each argument in *args*
            *weights*: {``{}``} | :class:`dict` (:class:`float` > 0)
                Dictionary of arg-specific distance weights
        :Outputs:
            *y*: :class:`float` | *db[col].__class__*
                Value of *db[col]* at point closest to *a*
        :Versions:
            * 2018-12-30 ``@ddalle``: Version 1.0
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
            * 2020-04-24 ``@ddalle``: Switched args to :class:`tuple`
            * 2020-05-19 ``@ddalle``: Support for 2D cols
        """
        # Check for column
        if (col not in self.cols) or (col not in self):
            # Missing col
            raise KeyError("Col '%s' is not present" % col)
        # Get values
        V = self.get_all_values(col)
        # Array length
        if isinstance(V, np.ndarray):
            # Use the last dimension
            n = V.shape[-1]
            # Get dimensionality
            ndim = V.ndim
        else:
            # Use a a simple length
            n = len(V)
            # Always 1D
            ndim = 1
        # Initialize distances
        d = np.zeros(n, dtype="float")
        # Dictionary of distance weights
        W = kw.get("weights", {})
        # Loop through keys
        for (i, k) in enumerate(args):
            # Get value
            xi = a[i]
            # Get weight
            wi = W.get(k, 1.0)
            # Distance
            d += wi*(self[k] - xi)**2
        # Find minimum distance
        j = np.argmin(d)
        # Perform interpolation
        if ndim == 1:
            # Single value
            return V[j]
        elif ndim == 2:
            # Use column *j*
            return V[:,j]

    # Evaluate UQ from coefficient
    def rcall_uq(self, *a, **kw):
        r"""Evaluate specified UQ cols for a specified col

        This function will evaluate the UQ cols specified for a given
        nominal column by referencing the appropriate subset of
        *db.response_args* for any UQ cols.  It evaluates the UQ col
        named in *db.uq_cols*.  For example if *CN* is a function of
        ``"mach"``, ``"alpha"``, and ``"beta"``; ``db.uq_cols["CN"]``
        is *UCN*; and *UCN* is a function of ``"mach"`` only, this
        function passes only the Mach numbers to *UCN* for evaluation.

        :Call:
            >>> U = db.rcall_uq(*a, **kw)
            >>> U = db.rcall_uq(col, x0, X1, ...)
            >>> U = db.rcall_uq(col, k0=x0, k1=x1, ...)
            >>> U = db.rcall_uq(col, k0=x0, k1=X1, ...)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of **nominal** column to evaluate
            *db.uq_cols*: :class:`dict`\ [:class:`str`]
                Dictionary of UQ col names for each col
            *x0*: :class:`float` | :class:`int`
                Numeric value for first argument to *col* evaluator
            *x1*: :class:`float` | :class:`int`
                Numeric value for second argument to *col* evaluator
            *X1*: :class:`np.ndarray`\ [:class:`float`]
                Array of *x1* values
            *k0*: :class:`str`
                Name of first argument to *col* evaluator
            *k1*: :class:`str`
                Name of second argument to *col* evaluator
        :Outputs:
            *U*: :class:`dict`\ [:class:`float` | :class:`np.ndarray`]
                Values of relevant UQ col(s) by name
        :Versions:
            * 2019-03-07 ``@ddalle``: Version 1.0
            * 2019-12-26 ``@ddalle``: From :mod:`tnakit`
        """
       # --- Get coefficient name ---
        # Process coefficient name and remaining coeffs
        col, a, kw = self._prep_args_colname(*a, **kw)
       # --- Argument processing ---
        # Specific lookup arguments
        args_col = self.get_response_args(col)
        # Initialize lookup point
        x = []
        # Loop through arguments
        for i, k in enumerate(args_col):
            # Get value
            xi = self.get_arg_value(i, k, *a, **kw)
            # Save it
            x.append(np.asarray(xi))
        # Normalize arguments
        X, dims = self.normalize_args(x)
        # Maximum dimension
        nd = len(dims)
       # --- UQ coeff ---
        # Dictionary of UQ coefficients
        uq_cols = getattr(self, "uq_cols", {})
        # Coefficients for this coefficient
        uq_col = uq_cols.get(col, [])
        # Check for list
        if isinstance(uq_col, (tuple, list)):
            # Save a flag for list of coeffs
            qscalar = False
            # Pass coefficient to list (copy it)
            uq_col_list = list(uq_col)
        else:
            # Save a flag for scalar output
            qscalar = True
            # Make a list
            uq_col_list = [uq_col]
       # --- Evaluation ---
        # Initialize output
        U = {}
        # Loop through UQ coeffs
        for uk in uq_col_list:
            # Get evaluation args
            args_k = self.get_response_args(uk)
            # Initialize inputs to *uk*
            UX = []
            # Loop through eval args
            for ai in args_k:
                # Check for membership
                if ai not in args_col:
                    raise ValueError(
                        ("UQ col '%s' is a function of " % uk) +
                        ("'%s', but parent col '%s' is not" % (ai, col)))
                # Append value
                UX.append(X[args_col.index(ai)])
            # Evaluate
            U[uk] = self.__call__(uk, *UX, **kw)

       # --- Output ---
        # Check for scalar output
        if qscalar:
            # Return first value
            return U[uk]
        else:
            # Return list
            return U

    # Evaluate coefficient from arbitrary list of arguments
    def rcall_from_arglist(self, col, args, *a, **kw):
        r"""Evaluate column from arbitrary argument list

        This function is used to evaluate a col when given the
        arguments to some other column.

        :Call:
            >>> V = db.rcall_from_arglist(col, args, *a, **kw)
            >>> V = db.rcall_from_arglist(col, args, x0, X1, ...)
            >>> V = db.rcall_from_arglist(col, args, k0=x0, k1=x1, ...)
            >>> V = db.rcall_from_arglist(col, args, k0=x0, k1=X1, ...)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *args*: :class:`list`\ [:class:`str`]
                List of arguments provided
            *x0*: :class:`float` | :class:`int`
                Numeric value for first argument to *col* evaluator
            *x1*: :class:`float` | :class:`int`
                Numeric value for second argument to *col* evaluator
            *X1*: :class:`np.ndarray`\ [:class:`float`]
                Array of *x1* values
            *k0*: :class:`str`
                Name of first argument to *col* evaluator
            *k1*: :class:`str`
                Name of second argument to *col* evaluator
        :Outputs:
            *V*: :class:`float` | :class:`np.ndarray`
                Values of *col* as appropriate
        :Versions:
            * 2019-03-13 ``@ddalle``: Version 1.0
            * 2019-12-26 ``@ddalle``: From :mod:`tnakit`
        """
       # --- Argument processing ---
        # Specific lookup arguments for *coeff*
        args_col = self.get_response_args(col)
        # Initialize lookup point
        x = []
        # Loop through arguments asgiven
        for i, k in enumerate(args):
            # Get value
            xi = self.get_arg_value(i, k, *a, **kw)
            # Save it
            x.append(np.asarray(xi))
        # Normalize arguments
        X, dims = self.normalize_args(x)
        # Maximum dimension
        nd = len(dims)
       # --- Evaluation ---
        # Initialize inputs to *coeff*
        A = []
        # Get aliases for this coeffiient
        aliases = getattr(self, "response_arg_aliases", {})
        aliases = aliases.get(col, {})
        # Loop through eval args
        for ai in args_col:
            # Check for membership
            if ai in args:
                # Append value
                A.append(X[args.index(ai)])
                continue
            # Check aliases
            for k, v in aliases.items():
                # Check if we found the argument sought for
                if v != ai:
                    continue
                # Check if this alias is in the provided list
                if k in args:
                    # Replacement argument name
                    ai = k
            # Check for membership (second try)
            if ai in args:
                # Append value
                A.append(X[args.index(ai)])
                continue
            raise ValueError(
                ("Col '%s' is a function of " % col) +
                ("'%s', not provided in argument list" % ai))
        # Evaluate
        return self.__call__(col, *A, **kw)

    # Evaluate coefficient from arbitrary list of arguments
    def rcall_from_index(self, col, I, **kw):
        r"""Evaluate data column from indices

        This function has the same output as accessing ``db[col][I]`` if
        *col* is directly present in the database.  However, it's
        possible that *col* can be evaluated by some other technique, in
        which case direct access would fail but this function may still
        succeed.

        This function looks up the appropriate input variables and uses
        them to generate inputs to the database evaluation method.

        :Call:
            >>> V = db.rcall_from_index(col, I, **kw)
            >>> v = db.rcall_from_index(col, i, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Indices at which to evaluate function
            *i*: :class:`int`
                Single index at which to evaluate
        :Outputs:
            *V*: :class:`np.ndarray`
                Values of *col* as appropriate
            *v*: :class:`float`
                Scalar evaluation of *col*
        :Versions:
            * 2019-03-13 ``@ddalle``: Version 1.0
            * 2019-12-26 ``@ddalle``: From :mod:`tnakit`
        """
       # --- Argument processing ---
        # Specific lookup arguments for *col*
        args_col = self.get_response_args(col)
       # --- Evaluation ---
        # Initialize inputs to *coeff*
        A = []
        # Loop through eval args
        for ai in args_col:
            # Append value
            A.append(self.get_xvals(ai, I, **kw))
        # Evaluate
        return self.__call__(col, *A, **kw)

   # --- Declaration ---
    # Set evaluation methods
    def make_responses(self, cols, method, args, *a, **kw):
        r"""Set evaluation method for a list of columns

        :Call:
            >>> db.make_responses(cols, method, args, *a, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *cols*: :class:`list`\ [:class:`str`]
                List of columns for which to declare evaluation rules
            *method*: ``"nearest"`` | ``"linear"`` | :class:`str`
                Response (lookup/interpolation/evaluation) method name
            *args*: :class:`list`\ [:class:`str`]
                List of input arguments
            *a*: :class:`tuple`
                Args passed to constructor, if used
            *aliases*: {``{}``} | :class:`dict`\ [:class:`str`]
                Dictionary of alternate variable names during
                evaluation; if *aliases[k1]* is *k2*, that means *k1*
                is an alternate name for *k2*, and *k2* is in *args*
            *response_kwargs*: {``{}``} | :class:`dict`
                Keyword arguments passed to functions
            *I*: {``None``} | :class:`np.ndarray`
                Indices of cases to include in response {all}
            *function*: {``"cubic"``} | :class:`str`
                Radial basis function type
            *smooth*: {``0.0``} | :class:`float` >= 0
                Smoothing factor for methods that allow inexact
                interpolation, ``0.0`` for exact interpolation
        :Versions:
            * 2019-01-07 ``@ddalle``: Version 1.0
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
            * 2020-02-18 ``@ddalle``: Name from :func:`SetEvalMethod`
            * 2020-03-06 ``@ddalle``: Name from :func:`set_responses`
        """
        # Check for list
        if not isinstance(cols, (list, tuple, set)):
            # Not a list
            raise TypeError(
                "Response col list must be list, " +
                ("got '%s'" % type(cols)))
        # Loop through coefficients
        for col in cols:
            # Check type
            if not isinstance(col, typeutils.strlike):
                # Not a string
                raise TypeError("Response col must be a string")
            # Specify individual col
            self.make_response(col, method, args, *a, **kw)

    # Save a method for one coefficient
    def make_response(self, col, method, args, *a, **kw):
        r"""Set evaluation method for a single column

        :Call:
            >>> db.make_response(col, method, args, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column for which to declare evaluation rules
            *method*: ``"nearest"`` | ``"linear"`` | :class:`str`
                Response (lookup/interpolation/evaluation) method name
            *args*: :class:`list`\ [:class:`str`]
                List of input arguments
            *a*: :class:`tuple`
                Args passed to constructor, if used
            *ndim*: {``0``} | :class:`int` >= 0
                Output dimensionality
            *aliases*: {``{}``} | :class:`dict`\ [:class:`str`]
                Dictionary of alternate variable names during
                evaluation; if *aliases[k1]* is *k2*, that means *k1*
                is an alternate name for *k2*, and *k2* is in *args*
            *response_kwargs*: {``{}``} | :class:`dict`
                Keyword arguments passed to functions
            *I*: {``None``} | :class:`np.ndarray`
                Indices of cases to include in response surface {all}
            *function*: {``"cubic"``} | :class:`str`
                Radial basis function type
            *smooth*: {``0.0``} | :class:`float` >= 0
                Smoothing factor for methods that allow inexact
                interpolation, ``0.0`` for exact interpolation
            *func*: **callable**
                Function to use for ``"function"`` *method*
            *extracols*: {``None``} | :class:`set` | :class:`list`
                Additional col names that might be used as kwargs
        :Versions:
            * 2019-01-07 ``@ddalle``: Version 1.0
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
            * 2019-12-30 ``@ddalle``: Version 2.0; map of methods
            * 2020-02-18 ``@ddalle``: Name from :func:`_set_method1`
            * 2020-03-06 ``@ddalle``: Name from :func:`set_response`
            * 2020-04-24 ``@ddalle``: Add *response_arg_alternates*
        """
       # --- Input checks ---
        # Check inputs
        if col is None:
            # Set the default
            col = "_"
        # Check for valid argument list
        if args is None:
            raise ValueError("Argument list (keyword 'args') is required")
        # Check for method
        if method is None:
            raise ValueError("Eval method (keyword 'method') is required")
        # Get alias option
        arg_aliases = kw.get("aliases", {})
        # Get alias option
        response_kwargs = kw.get("response_kwargs", {})
        # Save aliases
        self.set_response_arg_aliases(col, arg_aliases)
        self.set_response_kwargs(col, response_kwargs)
       # --- Method switch ---
        # Get class
        cls = self.__class__
        # Get dimension
        ndim = kw.get("ndim", self.get_output_ndim(col))
        # Set dimensionality (handles checks first)
        self.set_output_ndim(col, ndim)
        # Use lower case with hyphens instead of underscores
        method = method.lower().replace("_", "-")
        # Get proper method name (default to same)
        method = cls._method_map.get(method, method)
        # Check if present
        if method not in cls._method_names:
             # Get close matches
            mtchs = difflib.get_close_matches(method_col, cls._method_names)
            # Error message
            raise ValueError(
                ("No %i-D eval method '%s'; " % (ndim, method)) +
                ("closest matches: %s" % mtches))
        # Check for required constructor method
        constructor_name = cls._method_constructors.get(method)
        # Get constructor
        if constructor_name is not None:
            constructor_col = getattr(self, constructor_name, None)
        # Apply it if appropriate
        if constructor_name is None:
            # Do nothing
            pass
        elif not callable(constructor_col):
            raise TypeError(
                "Constructor for method '%s' is not callable" % method)
        else:
            # Call the constructor
            constructor_col(col, *a, args=args, **kw)
        # Save method name
        self.set_response_method(col, method)
        # Argument list is the same for all methods
        self.set_response_args(col, args)
        # Construct list (set actually) of kwargs for this key
        self.create_arg_alternates(col, extracols=kw.get("extracols"))

   # --- Constructors ---
    # Explicit function
    def _create_function(self, col, *a, **kw):
        r"""Constructor for ``"function"`` methods

        :Call:
            >>> db._create_function(col, *a, **kw)
            >>> db._create_function(col, fn, *a[1:], **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *fn*, *a[0]*: *callable*
                Function to save
            *a*: :class:`tuple`
                Extra positional arguments ignored
        :Keywords:
            *function*, *func*: *callable*
                Callable function to save, overrides *a[0]*
            *use_self*: {``True``} | ``False``
                Flag to include database in callback
        :Versions:
            * 2019-12-30 ``@ddalle``: Version 1.0
        """
        # Create response_funcs dictionary
        response_funcs = self.__dict__.setdefault("response_funcs", {})
        # Create response_funcs dictionary
        response_funcs_self = self.__dict__.setdefault(
            "response_funcs_self", {})
        # Get the function
        if len(a) > 0:
            # Function given as arg
            func = a[0]
        else:
            # Function better be a keyword because there are no args
            func = None
        # Get function
        func = kw.get("func", kw.get("function", func))
        # Save the function
        response_funcs[col] = func
        response_funcs_self[col] = kw.get("use_self", True)

    # Global RBFs
    def _create_rbf(self, col, *a, **kw):
        r"""Constructor for ``"rbf"`` methods

        :Call:
            >>> db._create_rbf(col, *a, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *a*: :class:`tuple`
                Extra positional arguments ignored
        :Keywords:
            *args*: :class:`list`\ [:class:`str`]
                List of evaluation arguments
        :Versions:
            * 2019-12-30 ``@ddalle``: Version 1.0
        """
        # Get arguments arg
        args = kw.pop("args", None)
        # Check types
        if args is None:
            raise ValueError("'args' keyword argument is required")
        elif not isinstance(args, list):
            raise TypeError("'args' list must be list (got %s)" % type(args))
        # Call function
        self.create_global_rbfs([col], args, **kw)

    # Linear-RBFs
    def _create_rbf_linear(self, col, *a, **kw):
        r"""Constructor for ``"rbf-linear"`` methods

        :Call:
            >>> db._create_rbf_linear(col, *a, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *a*: :class:`tuple`
                Extra positional arguments ignored
        :Keywords:
            *args*: :class:`list`\ [:class:`str`]
                List of evaluation arguments
        :Versions:
            * 2019-12-30 ``@ddalle``: Version 1.0
        """
        # Get arguments arg
        args = kw.pop("args", None)
        # Check types
        if args is None:
            raise ValueError("'args' keyword argument is required")
        elif not isinstance(args, list):
            raise TypeError("'args' list must be list (got %s)" % type(args))
        # Call function
        self.create_slice_rbfs([col], args, **kw)

    # Schedule-RBFs
    def _create_rbf_map(self, col, *a, **kw):
        r"""Constructor for ``"rbf-map"`` methods

        :Call:
            >>> db._create_rbf_map(col, *a, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *a*: :class:`tuple`
                Extra positional arguments ignored
        :Keywords:
            *args*: :class:`list`\ [:class:`str`]
                List of evaluation arguments
        :Versions:
            * 2019-12-30 ``@ddalle``: Version 1.0
        """
        # Get arguments arg
        args = kw.pop("args", None)
        # Check types
        if args is None:
            raise ValueError("'args' keyword argument is required")
        elif not isinstance(args, list):
            raise TypeError("'args' list must be list (got %s)" % type(args))
        # Call function
        self.create_slice_rbfs([col], args, **kw)

   # --- Options: Get ---
    # Get argument list
    def get_response_args(self, col, argsdef=None):
        r"""Get list of evaluation arguments

        :Call:
            >>> args = db.get_response_args(col, argsdef=None)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *argsdef*: {``None``} | :class:`list`\ [:class:`str`]
                Default arg list if none found in *db*
        :Outputs:
            *args*: :class:`list`\ [:class:`str`]
                List of parameters used to evaluate *col*
        :Versions:
            * 2019-03-11 ``@ddalle``: Forked from :func:`__call__`
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
            * 2020-03-26 ``@ddalle``: Added *argsdef*
            * 2020-04-21 ``@ddalle``: Rename from :func:`get_eval_args`
        """
        # Get overall handle
        response_args = self.__dict__.get("response_args", {})
        # Get option
        args_col = response_args.get(col)
        # Check for default
        if args_col is None:
            # Attempt to get a default
            args_col = response_args.get("_")
        # Create a copy if a list
        if args_col is None:
            # User user-provided default
            args_col = argsdef
        # Check type and make copy
        if args_col is None:
            # Not set up
            return
        elif isinstance(args_col, list):
            # Create a copy to prevent muting the definitions
            return list(args_col)
        else:
            # What?
            raise TypeError(
                "response_args for '%s' must be list (got %s)"
                % (col, type(args_col)))

    # Get evaluation method
    def get_response_method(self, col):
        r"""Get evaluation method (if any) for a column

        :Call:
            >>> method = db.get_response_method(col)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
        :Outputs:
            *method*: ``None`` | :class:`str`
                Name of evaluation method for *col* or ``"_"``
        :Versions:
            * 2019-03-13 ``@ddalle``: Version 1.0
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
            * 2019-12-30 ``@ddalle``: Added default
        """
        # Get attribute
        response_methods = self.__dict__.setdefault("response_methods", {})
        # Get method
        method = response_methods.get(col)
        # Check for ``None``
        if method is None:
            # Get default
            method = response_methods.get("_")
        # Output
        return method

    # Get evaluation argument converter
    def get_response_arg_converter(self, k):
        r"""Get evaluation argument converter

        :Call:
            >>> f = db.get_response_arg_converter(k)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *k*: :class:`str` | :class:`unicode`
                Name of argument
        :Outputs:
            *f*: ``None`` | callable
                Callable converter
        :Versions:
            * 2019-03-13 ``@ddalle``: Version 1.0
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Get converter dictionary
        converters = self.__dict__.setdefault("response_arg_covnerters", {})
        # Get converter
        f = converters.get(k)
        # Output if None
        if f is None:
            return f
        # Check class
        if not callable(f):
            raise TypeError("Converter for '%s' is not callable" % k)
        # Output
        return f

    # Get user-set callable function
    def get_response_func(self, col):
        r"""Get callable function predefined for a column

        :Call:
            >>> fn = db.get_response_func(col)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column to evaluate
        :Outputs:
            *fn*: ``None`` | *callable*
                Specified function for *col*
        :Versions:
            * 2019-12-28 ``@ddalle``: Version 1.0
        """
        # Get dictionary
        response_funcs = self.__dict__.get("response_funcs", {})
        # Check types
        if not typeutils.isstr(col):
            raise TypeError(
                "Data column name must be string (got %s)" % type(col))
        elif not isinstance(response_funcs, dict):
            raise TypeError("response_funcs attribute is not a dict")
        # Get entry
        fn = response_funcs.get(col)
        # If none, acceptable
        if fn is None:
            return
        # Check type if nonempty
        if not callable(fn):
            raise TypeError("response_func for col '%s' is not callable" % col)
        # Output
        return fn

    # Get aliases for evaluation args
    def get_response_arg_aliases(self, col):
        r"""Get alias names for evaluation args for a data column

        :Call:
            >>> aliases = db.get_response_arg_aliases(col)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column to evaluate
        :Outputs:
            *aliases*: {``{}``} | :class:`dict`
                Alternate names for args while evaluationg *col*
        :Versions:
            * 2019-12-30 ``@ddalle``: Version 1.0
        """
        # Get attribute
        arg_aliases = self.__dict__.get("response_arg_aliases", {})
        # Check types
        if not typeutils.isstr(col):
            raise TypeError(
                "Data column name must be string (got %s)" % type(col))
        elif not isinstance(arg_aliases, dict):
            raise TypeError("response_arg_aliases attribute is not a dict")
        # Get entry
        aliases = arg_aliases.get(col)
        # Check for empty response
        if aliases is None:
            # Use defaults
            aliases = arg_aliases.get("_", {})
        # Check types
        if not isinstance(aliases, dict):
            raise TypeError(
                "Aliases for col '%s' must be dict (got %s)" %
                (col, type(aliases)))
        # (Not checking key-value types)
        # Output
        return aliases

    # Get eval arg keywords
    def get_response_kwargs(self, col):
        r"""Get any keyword arguments passed to *col* evaluator

        :Call:
            >>> kwargs = db.get_response_kwargs(col)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column to evaluate
        :Outputs:
            *kwargs*: {``{}``} | :class:`dict`
                Keyword arguments to add while evaluating *col*
        :Versions:
            * 2019-12-30 ``@ddalle``: Version 1.0
        """
        # Get attribute
        response_kwargs = self.__dict__.get("response_kwargs", {})
        # Check types
        if not typeutils.isstr(col):
            raise TypeError(
                "Data column name must be string (got %s)" % type(col))
        elif not isinstance(response_kwargs, dict):
            raise TypeError("response_kwargs attribute is not a dict")
        # Get entry
        kwargs = response_kwargs.get(col)
        # Check for empty response
        if kwargs is None:
            # Use defaults
            kwargs = response_kwargs.get("_", {})
        # Check types
        if not isinstance(kwargs, dict):
            raise TypeError(
                "response_kwargs for col '%s' must be dict (got %s)" %
                (col, type(kwargs)))
        # Output
        return kwargs

    # Get xvars for output
    def get_output_xargs(self, col):
        r"""Get list of args to output for column *col*

        :Call:
            >>> xargs = db.get_output_xargs(col)
        :Inputs:
            *db*: :class:`cape.attdb.rdbscalar.DBResponseLinear`
                Database with multidimensional output functions
            *col*: :class:`str`
                Name of column to evaluate
        :Outputs:
            *xargs*: {``[]``} | :class:`list`\ [:class:`str`]
                List of input args to one condition of *col*
        :Versions:
            * 2019-12-30 ``@ddalle``: Version 1.0
            * 2020-03-27 ``@ddalle``: From *db.defns* to *db.response_xargs*
        """
        # Get attribute
        response_xargs = self.__dict__.get("response_xargs", {})
        # Get dimensionality
        xargs = response_xargs.get(col)
        # De-None
        if xargs is None:
            xargs = []
        # Check type
        if not isinstance(xargs, list):
            raise TypeError(
                "response_xargs for col '%s' must be list (got %s)"
                % (col, type(xargs)))
        # Output (copy)
        return list(xargs)

    def get_output_xarg1(self, col):
        r"""Get single arg for output for column *col*

        :Call:
            >>> xarg = db.get_output_xarg1(col)
        :Inputs:
            *db*: :class:`cape.attdb.rdbscalar.DBResponseLinear`
                Database with multidimensional output functions
            *col*: :class:`str`
                Name of column to evaluate
        :Outputs:
            *xarg*: ``None`` | :class:`str`
                Input arg to function for one condition of *col*
        :Versions:
            * 2021-12-16 ``@ddalle``: Version 1.0
        """
        # Get *xk* for output
        xargs = self.get_output_xargs(col)
        # Unpack
        if isinstance(xargs, list):
            # Check length
            if len(xargs) == 0:
                return
            else:
                return xargs[0]
        else:
            return

    # Get auxiliary cols
    def get_response_acol(self, col):
        r"""Get names of any aux cols related to primary col

        :Call:
            >>> acols = db.get_response_acol(col)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column to evaluate
        :Outputs:
            *acols*: :class:`list`\ [:class:`str`]
                Name of aux columns required to evaluate *col*
        :Versions:
            * 2020-03-23 ``@ddalle``: Version 1.0
            * 2020-04-21 ``@ddalle``: Rename *eval_acols*
        """
        # Get dictionary of ecols
        response_acols = self.__dict__.get("response_acols", {})
        # Get ecols
        acols = response_acols.get(col, [])
        # Check type
        if typeutils.isstr(acols):
            # Create single list
            acols = [acols]
        elif ecols is None:
            # Empty result should be empty list
            acols = []
        elif not isinstance(acols, list):
            # Invalid type
            raise TypeError(
                "response_acols for col '%s' should be list; got '%s'"
                % (col, type(acols)))
        # Return it
        return acols

   # --- Options: Set ---
    # Set evaluation args
    def set_response_args(self, col, args):
        r"""Set list of evaluation arguments for a column

        :Call:
            >>> db.set_response_args(col, args)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column
            *args*: :class:`list`\ [:class:`str`]
                List of arguments for evaluating *col*
        :Effects:
            *db.response_args*: :class:`dict`
                Entry for *col* set to copy of *args* w/ type checks
        :Versions:
            * 2019-12-28 ``@ddalle``: Version 1.0
            * 2020-04-21 ``@ddalle``: Rename from :func:`set_eval_args`
        """
        # Check types
        if not typeutils.isstr(col):
            raise TypeError(
                "Data column name must be str (got %s)" % type(col))
        if not isinstance(args, list):
            raise TypeError(
                "response_args for '%s' must be list (got %s)"
                % (col, type(args)))
        # Check args
        for (j, k) in enumerate(args):
            if not typeutils.isstr(k):
                raise TypeError(
                    "Arg %i for col '%s' is not a string" % (j, col))
        # Get handle to attribute
        response_args = self.__dict__.setdefault("response_args", {})
        # Check type
        if not isinstance(response_args, dict):
            raise TypeError("response_args attribute is not a dict")
        # Set parameter (to a copy)
        response_args[col] = list(args)

    # Set evaluation method
    def set_response_method(self, col, method):
        r"""Set name (only) of evaluation method

        :Call:
            >>> db.set_response_method(col, method)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column
            *method*: :class:`str`
                Name of evaluation method (only checked for type)
        :Effects:
            *db.response_methods*: :class:`dict`
                Entry for *col* set to *method*
        :Versions:
            * 2019-12-28 ``@ddalle``: Version 1.0
        """
        # Check types
        if not typeutils.isstr(col):
            raise TypeError(
                "Data column name must be str (got %s)" % type(col))
        if not typeutils.isstr(method):
            raise TypeError(
                "response_method for '%s' must be list (got %s)"
                % (col, type(method)))
        # Get handle to attribute
        response_methods = self.__dict__.setdefault("response_methods", {})
        # Check type
        if not isinstance(response_methods, dict):
            raise TypeError("response_method attribute is not a dict")
        # Set parameter (to a copy)
        response_methods[col] = method

    # Set evaluation function
    def set_response_func(self, col, fn):
        r"""Set specific callable for a column

        :Call:
            >>> db.set_response_func(col, fn)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column
            *fn*: *callable* | ``None``
                Function or other callable entity
        :Effects:
            *db.response_methods*: :class:`dict`
                Entry for *col* set to *method*
        :Versions:
            * 2019-12-28 ``@ddalle``: Version 1.0
        """
        # Check types
        if not typeutils.isstr(col):
            raise TypeError(
                "Data column name must be str (got %s)" % type(col))
        if (fn is not None) and not callable(fn):
            raise TypeError(
                "response_func for '%s' must be callable" % col)
        # Get handle to attribute
        response_funcs = self.__dict__.setdefault("response_funcs", {})
        # Check type
        if not isinstance(response_funcs, dict):
            raise TypeError("response_funcs attribute is not a dict")
        # Set parameter
        if fn is None:
            # Remove it
            response_funcs.pop(col, None)
        else:
            # Set it
            response_funcs[col] = fn

    # Set a default value for an argument
    def set_arg_default(self, k, v):
        r"""Set a default value for an evaluation argument

        :Call:
            >>> db.set_arg_default(k, v)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *k*: :class:`str`
                Name of evaluation argument
            *v*: :class:`float`
                Default value of the argument to set
        :Versions:
            * 2019-02-28 ``@ddalle``: Version 1.0
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Get dictionary
        arg_defaults = self.__dict__.setdefault("response_arg_defaults", {})
        # Save key/value
        arg_defaults[k] = v

    # Set a conversion function for input variables
    def set_arg_converter(self, k, fn):
        r"""Set a function to evaluation argument for a specific argument

        :Call:
            >>> db.set_arg_converter(k, fn)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *k*: :class:`str`
                Name of evaluation argument
            *fn*: :class:`function`
                Conversion function
        :Versions:
            * 2019-02-28 ``@ddalle``: Version 1.0
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Check input
        if not callable(fn):
            raise TypeError("Converter is not callable")
        # Get dictionary of converters
        arg_converters = self.__dict__.setdefault("response_arg_converters", {})
        # Save function
        arg_converters[k] = fn

    # Set eval argument aliases
    def set_response_arg_aliases(self, col, aliases):
        r"""Set alias names for evaluation args for a data column

        :Call:
            >>> db.set_response_arg_aliases(col, aliases)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column to evaluate
            *aliases*: {``{}``} | :class:`dict`
                Alternate names for args while evaluationg *col*
        :Versions:
            * 2019-12-30 ``@ddalle``: Version 1.0
        """
        # Transform any False-like thing to {}
        if not aliases:
            aliases = {}
        # Get attribute
        arg_aliases = self.__dict__.setdefault("response_arg_aliases", {})
        # Check types
        if not typeutils.isstr(col):
            raise TypeError(
                "Data column name must be string (got %s)" % type(col))
        elif not isinstance(arg_aliases, dict):
            raise TypeError("response_arg_aliases attribute is not a dict")
        elif not isinstance(aliases, dict):
            raise TypeError(
                "aliases arg must be dict (got %s)" % type(aliases))
        # Check key-value types
        for (k, v) in aliases.items():
            # Check key
            if not typeutils.isstr(k):
                raise TypeError(
                    "Found alias key for '%s' that is not a string" % col)
            if not typeutils.isstr(v):
                raise TypeError(
                    "Alias for '%s' in col '%s' is not a string" % (k, col))
        # Save it
        arg_aliases[col] = aliases

    # Set eval argument keyword arguments
    def set_response_kwargs(self, col, kwargs):
        r"""Set evaluation keyword arguments for *col* evaluator

        :Call:
            >>> db.set_response_kwargs(col, kwargs)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column to evaluate
            *kwargs*: {``{}``} | :class:`dict`
                Keyword arguments to add while evaluating *col*
        :Versions:
            * 2019-12-30 ``@ddalle``: Version 1.0
        """
        # Transform any False-like thing to {}
        if not kwargs:
            kwargs = {}
        # Get attribute
        response_kwargs = self.__dict__.setdefault("response_kwargs", {})
        # Check types
        if not typeutils.isstr(col):
            raise TypeError(
                "Data column name must be string (got %s)" % type(col))
        elif not isinstance(response_kwargs, dict):
            raise TypeError("response_kwargs attribute is not a dict")
        elif not isinstance(kwargs, dict):
            raise TypeError(
                "kwargs must be dict (got %s)" % type(kwargs))
        # Check key-value types
        for (k, v) in kwargs.items():
            # Check key
            if not typeutils.isstr(k):
                raise TypeError(
                    "Found keyword for '%s' that is not a string" % col)
        # Save it
        response_kwargs[col] = kwargs

    # Set xargs for output
    def set_output_xargs(self, col, xargs):
        r"""Set list of args to output for column *col*

        :Call:
            >>> db.set_output_xargs(col, xargs)
        :Inputs:
            *db*: :class:`cape.attdb.rdbscalar.DBResponseLinear`
                Database with multidimensional output functions
            *col*: :class:`str`
                Name of column to evaluate
            *xargs*: :class:`list`\ [:class:`str`]
                List of input args to one condition of *col*
        :Versions:
            * 2019-12-30 ``@ddalle``: Version 1.0
            * 2020-03-27 ``@ddalle``: From *db.defns* to *db.response_xargs*
        """
        # De-None
        if xargs is None:
            xargs = []
        # Get attribute
        response_xargs = self.__dict__.setdefault("response_xargs", {})
        # Check type
        if not isinstance(xargs, list):
            raise TypeError(
                "OutputXVars for col '%s' must be list (got %s)"
                % (col, type(xargs)))
        # Check contents
        for (j, k) in enumerate(xargs):
            if not typeutils.isstr(k):
                raise TypeError(
                    "Output arg %i for col '%s' must be str (got %s)"
                    % (j, col, type(k)))
        # Set (copy)
        response_xargs[col] = list(xargs)

    # Get auxiliary cols
    def set_response_acol(self, col, acols):
        r"""Set names of any aux cols related to primary col

        :Call:
            >>> db.set_response_acol(col, acols)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column to evaluate
            *acols*: :class:`list`\ [:class:`str`]
                Name of aux columns required to evaluate *col*
        :Versions:
            * 2020-03-23 ``@ddalle``: Version 1.0
            * 2020-04-21 ``@ddalle``: Rename *eval_acols*
        """
        # Check type
        if typeutils.isstr(acols):
            # Create single list
            acols = [acols]
        elif acols is None:
            # Empty result should be empty list
            acols = []
        elif not isinstance(acols, list):
            # Invalid type
            raise TypeError(
                "response_acols for col '%s' should be list; got '%s'"
                % (col, type(acols)))
        # Get dictionary of ecols
        response_acols = self.__dict__.setdefault("response_acols", {})
        # Set it
        response_acols[col] = acols

   # --- Aliases and Tagcols ---
    # Create set of kwargs that might be used as alternates
    def create_arg_alternates(self, col, extracols=None):
        r"""Create set of keys that might be used as kwargs to *col*

        :Call:
            >>> db.create_arg_alternates(col, extracols=None)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column with response method
            *extracols*: {``None``} | :class:`set` | :class:`list`
                Additional col names that might be used as kwargs
        :Effects:
            *db.respone_arg_alternates[col]*: :class:`set`
                Cols that are used by response for *col*
        :Versions:
            * 2020-04-24 ``@ddalle``: Version 1.0
        """
        # Handle to class
        cls = self.__class__
        # Class attributes
        _tagcols = cls.__dict__.get("_tagcols", {})
        _tagsubs = cls.__dict__.get("_tagsubmap", {})
        # Initialize set
        args_alt = set(cls._tagsubcols.get(col, set()))
        # Use extra columns provided by user
        if extracols:
            args_alt.update(set(extracols))
        # Get list of args for *col*
        args = self.get_response_args(col)
        # Get aliases
        arg_aliases = self.get_response_arg_aliases(col)
        # Add any aliases
        if arg_aliases:
            # Loop through aliases
            for k1, k2 in arg_aliases.items():
                # Check if *k2* is an arg
                if k2 in args:
                    args.append(k1)
        # Loop through the response args
        for arg in args:
            # Get tag for this argument
            tag = self.get_col_prop(arg, "Tag", arg)
            # Get suggested cols for main arg
            cols = _tagcols.get(tag)
            # Join cols if possible
            if cols:
                args_alt.update(cols)
            # Other tags that might be used to compute this tag
            subtags = _tagsubs.get(tag)
            # Move on if no subtags
            if subtags is None:
                continue
            # Loop through them
            for tag in subtags:
                # Get cols that could be used to compute this tag
                cols = _tagcols.get(tag)
                # Join if possible
                if cols:
                    args_alt.update(cols)
        # Save them
        self.response_arg_alternates[col] = args_alt

    # Get alternate args
    def get_arg_alternates(self, col):
        r"""Get :class:`set` of usable keyword args for *col*

        :Call:
            >>> altcols = db.get_arg_alternates(col)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column with response method
        :Outputs:
            *altcols*: :class:`set`\ [:class:`str`\
                Cols that are used by response for *col*
        :Versions:
            * 2020-04-24 ``@ddalle``: Version 1.0
        """
        # Get dictionary
        arg_alts = self.__dict__.get("response_arg_alternates", {})
        # Return values for *col*
        return arg_alts.get(col, set())

    # Check mode for __call__ (either by index or response)
    def _check_callmode(self, col, *a, **kw):
        r"""Determine call mode

        :Call:
            >>> mode = db._check_callmode(col, *a, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column to look up or calculate
            *a*: :class:`tuple`
                Positional args to :func:`__call__`
            *kw*: :class:`dict`
                Keyword args to :func:`__call__` or other methods
        :Outputs:
            *mod*: ``0`` | ``1`` | ``2``
                Lookup method:
                    * ``0``: return all values
                    * ``1``: lookup by index
                    * ``2``: use declared response method
                    * ``3``: use :func:`rcall_exact`

        :Versions:
            * 2020-04-24 ``@ddalle``: Version 1.0
        """
        # Get method, if any
        method = self.get_response_method(col)
        # Get args
        args = self.get_response_args(col)
        # Number of expected args
        if args:
            narg = len(args)
        else:
            narg = 0
        # Number of positional args given
        na = len(a)
        nx = na
        # Check for all args specified
        if na >= narg > 1:
            # Sufficient args for response
            if method:
                # Use declared response
                return 2
            else:
                # Args declared but no method
                return 3
        # Otherwise, check for kwargs
        altargs = self.get_arg_alternates(col)
        # Loop through keywords
        for k in kw:
            # Check if it's a usable keyword arg
            if k in altargs:
                # Increase number of args
                nx += 1
        # Recheck
        if nx == 0:
            # No args at all
            return 0
        elif (na == 1) and self.check_mask(a[0]):
            # Given indices/mask in first arg
            return 1
        elif method:
            # Use declared response
            return 2
        elif narg > 0:
            # Args declared but no method
            return 3

    # Separate rcall keywords and other kwargs
    def sep_response_kwargs(self, col, **kw):
        r"""Separate kwargs used for response and other options

        :Call:
            >>> kwr, kwo = db.sep_response_kwargs(col, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column to look up or calculate
            *kw*: :class:`dict`
                Keyword args to :func:`__call__` or other methods
        :Outputs:
            *kwr*: :class:`dict`
                Keyword args to :func:`__call__` or other methods
        :Versions:
            * 2020-04-24 ``@ddalle``: Version 1.0
        """
        # Check for trivial case
        if not kw:
            # No kwargs to separate
            return {}, {}
        # Otherwise, check for kwargs
        altargs = self.get_arg_alternates(col)
        # Get nominal args
        args = self.get_response_args(col)
        # Check if any defined
        if (not args) and (not altargs):
            # All kwargs are "other"
            return {}, kw
        # Initialize groups
        kwr = {}
        kwo = {}
        # Loop through input kwargs
        for k, v in kw.items():
            # Check if it's an arg or possible arg
            if args and (k in args):
                # Main rcall() arg
                kwr[k] = v
            elif altargs and (k in altargs):
                # Alternate rcall arg
                kwr[k] = v
            else:
                # Other kwarg
                kwo[k] = v
        # Output
        return kwr, kwo

   # --- Arguments ---
    # Get argument value
    def get_arg_value(self, i, k, *a, **kw):
        r"""Get the value of the *i*\ th argument to a function

        :Call:
            >>> v = db.get_arg_value(i, k, *a, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *i*: :class:`int`
                Argument index within *db.response_args*
            *k*: :class:`str`
                Name of evaluation argument
            *a*: :class:`tuple`
                Arguments to :func:`__call__`
            *kw*: :class:`dict`
                Keyword arguments to :func:`__call__`
        :Outputs:
            *v*: :class:`float` | :class:`np.ndarray`
                Value of the argument, possibly converted
        :Versions:
            * 2019-02-28 ``@ddalle``: Version 1.0
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Number of direct arguments
        na = len(a)
        # Converters
        arg_converters = self.__dict__.get("response_arg_converters", {})
        arg_defaults   = self.__dict__.get("response_arg_defaults",   {})
        # Check for sufficient non-keyword inputs
        if na > i:
            # Directly specified
            xi = kw.get(k, a[i])
        else:
            # Get from keywords
            xi = kw.get(k)
        # In most cases, this is sufficient
        if xi is not None:
            return xi
        # Check for a converter
        fk = arg_converters.get(k)
        # Apply converter
        if fk:
            # Apply converters
            try:
                # Convert values
                try:
                    # Use args and kwargs
                    xi = fk(*x, **kw)
                except Exception:
                    # Use just kwargs
                    xi = fk(**kw)
                # Save it
                if xi is not None: return xi
            except Exception:
                # Function failure
                print("Eval argument converter for '%s' failed" % k)
        # Get default
        xi = arg_defaults.get(k)
        # Final check
        if xi is None:
            # No value determined
            raise ValueError(
                "Could not determine value for argument '%s'" % k)
        else:
            # Final output
            return xi

    # Get dictionary of argument values
    def get_arg_value_dict(self, *a, **kw):
        r"""Return a dictionary of normalized argument variables

        Specifically, the dictionary contains a key for every argument
        used to evaluate the coefficient that is either the first
        argument or uses the keyword argument *col*.

        :Call:
            >>> X = db.get_arg_value_dict(*a, **kw)
            >>> X = db.get_arg_value_dict(col, x1, x2, ..., k3=x3)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column
            *x1*: :class:`float` | :class:`np.ndarray`
                Value(s) of first argument
            *x2*: :class:`float` | :class:`np.ndarray`
                Value(s) of second argument, if applicable
            *k3*: :class:`str`
                Name of third argument or optional variant
            *x3*: :class:`float` | :class:`np.ndarray`
                Value(s) of argument *k3*, if applicable
        :Outputs:
            *X*: :class:`dict`\ [:class:`np.ndarray`]
                Dictionary of values for each key used to evaluate *col*
                according to *b.response_args[col]*; each entry of *X*
                will have the same size
        :Versions:
            * 2019-03-12 ``@ddalle``: Version 1.0
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
        """
       # --- Get column name ---
        # Col name should be either a[0] or kw["coeff"]
        col, a, kw = self._prep_args_colname(*a, **kw)
       # --- Argument processing ---
        # Specific lookup arguments
        args_col = self.get_response_args(col)
        # Initialize lookup point
        x = []
        # Loop through arguments
        for i, k in enumerate(args_col):
            # Get value
            xi = self.get_arg_value(i, k, *a, **kw)
            # Save it
            x.append(np.asarray(xi))
        # Normalize arguments
        xn, dims = self.normalize_args(x)
       # --- Output ---
        # Initialize
        X = {}
        # Loop through args
        for i, k in enumerate(args_col):
            # Save value
            X[k] = xn[i]
        # Output
        return X

    # Process coefficient name
    def _prep_args_colname(self, *a, **kw):
        r"""Process coefficient name from arbitrary inputs

        :Call:
            >>> col, a, kw = db._prep_args_colname(*a, **kw)
            >>> col, a, kw = db._prep_args_colname(col, *a, **kw)
            >>> col, a, kw = db._prep_args_colname(*a, col=c, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of evaluation col
            *a*: :class:`tuple`
                Other sequential inputs
            *kw*: :class:`dict`
                Other keyword inputs
        :Outputs:
            *col*: :class:`str`
                Name of column to evaluate, from *a[0]* or *kw["col"]*
            *a*: :class:`tuple`
                Remaining inputs with coefficient name removed
            *kw*: :class:`dict`
                Keyword inputs with coefficient name removed
        :Versions:
            * 2019-03-12 ``@ddalle``: Version 1.0
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
            * 2019-12-18 ``@ddalle``: From :func:`_process_coeff`
        """
        # Check for keyword
        col = kw.pop("col", None)
        # Check for string
        if typeutils.isstr(col):
            # Output
            return col, a, kw
        # Number of direct inputs
        na = len(a)
        # Process *coeff* from *a* if possible
        if na > 0:
            # First argument is coefficient
            col = a[0]
            # Check for string
            if typeutils.isstr(col):
                # Remove first entry
                a = a[1:]
                # Output
                return col, a, kw
        # Must be string-like
        raise TypeError("Column name must be a string")

    # Normalize arguments
    def normalize_args(self, x, asarray=False):
        r"""Normalized mixed float and array arguments

        :Call:
            >>> X, dims = db.normalize_args(x, asarray=False)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *x*: :class:`list`\ [:class:`float` | :class:`np.ndarray`]
                Values for arguments, either float or array
            *asarray*: ``True`` | {``False``}
                Force array output (otherwise allow scalars)
        :Outputs:
            *X*: :class:`list`\ [:class:`float` | :class:`np.ndarray`]
                Normalized arrays/floats all with same size
            *dims*: :class:`tuple`\ [:class:`int`]
                Original dimensions of non-scalar input array
        :Versions:
            * 2019-03-11 ``@ddalle``: Version 1.0
            * 2019-03-14 ``@ddalle``: Added *asarray* input
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
            * 2019-12-18 ``@ddalle``: Removed ``@staticmethod``
        """
        # Input size by argument
        nxi = [xi.size for xi in x]
        ndi = [xi.ndim for xi in x]
        # Maximum size
        nx = np.max(nxi)
        nd = np.max(ndi)
        # Index of maximum size
        ix = nxi.index(nx)
        # Corresponding shape
        dims = x[ix].shape
        # Check for forced array output
        if asarray and (nd == 0):
            # Ensure 1D output
            nd = 1
            dims = (1,)
        # Initialize final arguments
        X = []
        # Loop through arguments again
        for (i, xi) in enumerate(x):
            # Check for trivial case
            if nd == 0:
                # Save scalar
                X.append(xi)
                continue
            # Get sizes
            nxk = nxi[i]
            ndk = ndi[i]
            # Check for expansion
            if ndk == 0:
                # Scalar to array
                X.append(np.full(nx, xi))
            elif ndk != nd:
                # Inconsistent size
                raise ValueError(
                    "Cannot normalize %iD and %iD inputs" % (ndk, nd))
            elif nxk != nx:
                # Inconsistent size
                raise IndexError(
                    "Cannot normalize inputs with size %i and %i" % (nxk, nx))
            else:
                # Already array
                X.append(xi.flatten())
        # Output
        return X, dims

   # --- Breakpoint Schedule ---
    # Return break points for schedule
    def get_schedule(self, args, x, extrap=True):
        r"""Get lookup points for interpolation scheduled by master key

        This is a utility that is used for situations where the break
        points of some keys may vary as a schedule of another one.
        For example if the maximum angle of attack in the database is
        different at each Mach number.  This utility provides the
        appropriate point at which to interpolate the remaining keys
        at the value of the first key both above and below the input
        value.  The first argument, ``args[0]``, is the master key
        that controls the schedule.

        :Call:
            >>> i0, i1, f, x0, x1 = db.get_schedule(args, x, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *args*: :class:`list`\ [:class:`str`]
                List of input argument names (*args[0]* is master key)
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
            *extrap*: {``True``} | ``False``
                If ``False``, raise error when lookup value is outside
                break point range for any key at any slice
        :Outputs:
            *i0*: ``None`` | :class:`int`
                Lower bound index, if ``None``, extrapolation below
            *i1*: ``None`` | :class:`int`
                Upper bound index, if ``None``, extrapolation above
            *f*: 0 <= :class:`float` <= 1
                Lookup fraction, ``1.0`` if *v* is at upper bound
            *x0*: :class:`np.ndarray`\ [:class:`float`]
                Evaluation values for ``args[1:]`` at *i0*
            *x1*: :class:`np.ndarray`\ [:class:`float`]
                Evaluation values for ``args[1:]`` at *i1*
        :Versions:
            * 2019-04-19 ``@ddalle``: Version 1.0
            * 2019-07-26 ``@ddalle``: Vectorized
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Number of args
        narg = len(args)
        # Error check
        if narg < 2:
            raise ValueError("At least two args required for scheduled lookup")
        # Flag for array or scalar
        qvec = False
        # Number of points in arrays (if any)
        n = None
        # Loop through args
        for i, k in enumerate(args):
            # Get value
            V = x[i]
            # Check type
            if typeutils.isarray(V):
                # Turn on array flag
                qvec = True
                # Get size
                nk = len(V)
                # Check consistency
                if n is None:
                    # New size
                    n = nk
                elif nk != n:
                    # Inconsistent size
                    raise ValueError(
                        "Eval arg '%s' has size %i, expected %i" % (k, nk, n))
            elif not isinstance(V, (float, int, np.ndarray)):
                # Improper type
                raise TypeError(
                    "Eval arg '%s' has type '%s'" % (k, type(V)))
        # Check for arrays
        if not qvec:
            # Call scalar version
            return self._get_schedule(args, x, extrap=extrap)
        # Initialize tuple of fixed-size lookup points
        X = tuple()
        # Loop through args again
        for i, k in enumerate(args):
            # Get value
            V = x[i]
            # Check type
            if isinstance(V, (float, int)):
                # Create constant-value array
                X += (np.full(n, V),)
            else:
                # Copy array
                X += (V,)
        # Otherwise initialize arrays
        I0 = np.zeros(n, dtype="int")
        I1 = np.zeros(n, dtype="int")
        F  = np.zeros(n)
        # Initialize tuples of modified lookup points
        X0 = tuple([np.zeros(n) for i in range(narg-1)])
        X1 = tuple([np.zeros(n) for i in range(narg-1)])
        # Loop through points
        for j in range(n):
            # Get lookup points
            xj = tuple([X[i][j] for i in range(narg)])
            # Evaluations
            i0, i1, f, x0, x1 = self._get_schedule(
                list(args), xj, extrap=extrap)
            # Save indices
            I0[j] = i0
            I1[j] = i1
            # Save lookup fraction
            F[j] = f
            # Save modified lookup points
            for i in range(narg-1):
                X0[i][j] = x0[i]
                X1[i][j] = x1[i]
        # Output
        return I0, I1, F, X0, X1

    # Return break points for schedule
    def _get_schedule(self, args, x, extrap=True):
        r"""Get lookup points for interpolation scheduled by master key

        This is a utility that is used for situations where the break
        points of some keys may vary as a schedule of another one.
        For example if the maximum angle of attack in the database is
        different at each Mach number.  This utility provides the
        appropriate point at which to interpolate the remaining keys
        at the value of the first key both above and below the input
        value.  The first argument, ``args[0]``, is the master key
        that controls the schedule.

        :Call:
            >>> i0, i1, f, x0, x1 = db.get_schedule(args, x, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *args*: :class:`list`\ [:class:`str`]
                List of input argument names (*args[0]* is master key)
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
            *extrap*: {``True``} | ``False``
                If ``False``, raise error when lookup value is outside
                break point range for any key at any slice
        :Outputs:
            *i0*: ``None`` | :class:`int`
                Lower bound index, if ``None``, extrapolation below
            *i1*: ``None`` | :class:`int`
                Upper bound index, if ``None``, extrapolation above
            *f*: 0 <= :class:`float` <= 1
                Lookup fraction, ``1.0`` if *v* is at upper bound
            *x0*: :class:`np.ndarray`\ [:class:`float`]
                Evaluation values for ``args[1:]`` at *i0*
            *x1*: :class:`np.ndarray`\ [:class:`float`]
                Evaluation values for ``args[1:]`` at *i1*
        :Versions:
            * 2019-04-19 ``@ddalle``: Version 1.0
        """
        # Error check
        if len(args) < 2:
            raise ValueError("At least two args required for scheduled lookup")
        # Slice/scheduling key
        skey = args.pop(0)
        # Lookup value for first variable
        i0, i1, f = self.get_bkpt_index(skey, x[0])
        # Number of additional args
        narg = len(args)
        # Initialize lookup points at slice *i0* and slice *i1*
        x0 = np.zeros(narg)
        x1 = np.zeros(narg)
        # Loop through arguments
        for j, k in enumerate(args):
            # Get min and max values
            try:
                # Try the case of varying break points indexed to *skey*
                xmin0 = self.get_bkpt(k, i0, 0)
                xmin1 = self.get_bkpt(k, i1, 0)
                xmax0 = self.get_bkpt(k, i0, -1)
                xmax1 = self.get_bkpt(k, i1, -1)
            except TypeError:
                # Fixed break points (apparently)
                xmin0 = self.get_bkpt(k, 0)
                xmin1 = self.get_bkpt(k, 0)
                xmax0 = self.get_bkpt(k, -1)
                xmax1 = self.get_bkpt(k, -1)
            # Interpolate to current *skey* value
            xmin = (1-f)*xmin0 + f*xmin1
            xmax = (1-f)*xmax0 + f*xmax1
            # Get progress fraction at current inter-slice *skey* value
            if xmax - xmin < 1e-8:
                fj = 0.0
            else:
                fj = (x[j+1] - xmin) / (xmax-xmin)
            # Check for extrapolation
            if not extrap and ((fj < -1e-3) or (fj - 1 > 1e-3)):
                # Raise extrapolation error
                print("Extrapolation dectected:")
                print("  arg 0: %s=%.4e" % (skey, x[0]))
                for j1, k1 in enumerate(args):
                    print("  arg %i: %s=%.4e" % (j1+1, k1, x[j1+1]))
                sys.tracebacklimit = 2
                raise ValueError(
                    ("Value %.2e " % x[j+1]) +
                    ("for arg %i (%s) is outside " % (j, k)) +
                    ("bounds [%.2e, %.2e]" % (xmin, xmax)))
            # Lookup points at slices *i0* and *i1* using this prog frac
            x0[j] = (1-fj)*xmin0 + fj*xmax0
            x1[j] = (1-fj)*xmin1 + fj*xmax1
        # Output
        return i0, i1, f, x0, x1

   # --- Linear ---
    # Multilinear lookup
    def rcall_multilinear(self, col, args, *x, **kw):
        r"""Perform linear interpolation in *n* dimensions

        This assumes the database is ordered with the first entry of
        *args* varying the most slowly and that the data is perfectly
        regular.

        :Call:
            >>> y = db.rcall_multilinear(col, args, *x)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
            *bkpt*: ``True`` | {``False``}
                Flag to interpolate break points instead of data
        :Outputs:
            *y*: ``None`` | :class:`float` | ``db[col].__class__``
                Interpolated value from ``db[col]``
        :Versions:
            * 2018-12-30 ``@ddalle``: Version 1.0
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Call root method without two of the options
        return self._rcall_multilinear(col, args, x, **kw)

    # Evaluate multilinear interpolation with caveats
    def _rcall_multilinear(self, col, args, x, I=None, j=None, **kw):
        r"""Perform linear interpolation in *n* dimensions

        This assumes the database is ordered with the first entry of
        *args* varying the most slowly and that the data is perfectly
        regular.

        :Call:
            >>> y = db._rcall_multilinear(col, args, *x, I=None, j=None)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`list` | :class:`tuple` | :class:`np.ndarray`
                Vector of values for each argument in *args*
            *I*: {``None``} | :class:`np.ndarray`\ [:class:`int`]
                Optional subset of database on which to perform
                interpolation
            *j*: {``None``} | :class:`int`
                Slice index, used by :func:`rcall_multilinear_schedule`
            *bkpt*: ``True`` | {``False``}
                Flag to interpolate break points instead of data
        :Outputs:
            *y*: ``None`` | :class:`float` | ``DBc[coeff].__class__``
                Interpolated value from ``DBc[coeff]``
        :Versions:
            * 2018-12-30 ``@ddalle``: Version 1.0
            * 2019-04-19 ``@ddalle``: Moved from :func:`eval_multilnear`
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Check for break-point evaluation flag
        bkpt = kw.get("bkpt", kw.get("breakpoint", False))
        # Extrapolation option
        extrap = kw.get("extrap", "hold")
        # Possible values
        try:
            # Extract coefficient
            if bkpt:
                # Lookup from breakpoints
                V = self.bkpts[col]
            else:
                # Lookup from main data
                V = self[col]
        except KeyError:
            # Missing key
            raise KeyError("Col '%s' is not present" % col)
        # Subset if appropriate
        if I is not None:
            # Attempt to subset
            try:
                # Select some indices
                V = V[I]
            except Exception:
                # Informative error
                raise ValueError(
                    "Failed to subset col '%s' using class '%s'"
                    % (coeff, I.__class__))
        # Number of keys
        nk = len(args)
        # Dimension
        ndim = V.ndim
        # Check it
        if ndim not in [1, 2]:
            raise ValueError(
                "Col '%s' must have dimension 1 or 2; got %i" % (col, ndim))
        # Count
        n = V.shape[-1]
        # Get break points for this schedule
        bkpts = {}
        for k in args:
            bkpts[k] = self._scheduled_bkpts(k, j)
        # Lengths for each variable
        N = [len(bkpts[k]) for k in args]
        # Check consistency
        if np.prod(N) != n:
            raise ValueError(
                ("Column '%s' has size %i, " % (col, n)),
                ("but total size of args %s is %i." % (args, np.prod(N))))
        # Initialize list of indices for each key
        I0 = []
        I1 = []
        F1 = []
        # Get lookup indices for each argument
        for (i, k) in enumerate(args):
            # Lookup value
            xi = x[i]
            # Values
            Vk = bkpts[k]
            # Get indices
            i0, i1, f = self._bkpt_index(Vk, xi)
            # Check for problems
            if i0 is None:
                # Below
                # Above bounds
                if extrap in ["hold", "holdlast", "last"]:
                    # Hold last value
                    i0, i1 = i1, i1 + 1
                    f = 0.0
                elif extrap in ["linear"]:
                    # Just use last interval (*f* already computed)
                    i0, i1 = i1, i1 + 1
                else:
                    # No extrapolation
                    raise ValueError(
                        ("Value %s=%.4e " % (k, xi)) +
                        ("below lower bound (%.4e)" % Vk[0]))
            elif i1 is None:
                # Above bounds
                if extrap in ["hold", "holdlast", "last"]:
                    # Hold last value
                    i0, i1 = i0 - 1, i0
                    f = 1.0
                elif extrap in ["linear"]:
                    # Just use last interval (*f* already computed)
                    i0, i1 = i0 - 1, i0
                else:
                    # No extrapolation
                    raise ValueError(
                        ("Value %s=%.4e " % (k, xi)) +
                        ("above upper bound (%.4e)" % Vk[-1]))
            # Save values
            I0.append(i0)
            I1.append(i1)
            F1.append(f)
        # Index of the lowest corner
        j0 = 0
        # Loop through the keys
        for i in range(nk):
            # Get value
            i0 = I0[i]
            # Overall ; multiply index by size of remaining block
            j0 += i0 * int(np.prod(N[i+1:]))
        # Initialize overall indices and weights
        J = j0 * np.ones(2**nk, dtype="int")
        F = np.ones(2**nk)
        # Counter from 0 to 2^nk-1
        E = np.arange(2**nk)
        # Loop through keys again
        for i in range(nk):
            # Exponent of two to use for this key
            e = nk - i
            # Up or down for each of the 2^nk individual lookup points
            jupdown = E % 2**e // 2**(e-1)
            # Size of remaining block
            subblock = int(np.prod(N[i+1:]))
            # Increment overall indices
            J += jupdown*subblock
            # Progress fraction for this variable
            fi = F1[i]
            # Convert up/down to either fi or 1-fi
            Fi = (1-fi)*(1-jupdown) + jupdown*fi
            # Apply weights
            F *= Fi
        # Perform interpolation
        if ndim == 1:
            # Regular weighted sum of scalars
            return np.sum(F*V[J])
        elif ndim == 2:
            # Weighted dot product (of columns)
            return np.dot(V[:,J], F)

   # --- Multilinear-schedule ---
    # Multilinear lookup at each value of arg
    def rcall_multilinear_schedule(self, col, args, *x, **kw):
        r"""Perform "scheduled" linear interpolation in *n* dimensions

        This assumes the database is ordered with the first entry of
        *args* varying the most slowly and that the data is perfectly
        regular.  However, each slice at a constant value of *args[0]*
        may have separate break points for all the other args.  For
        example, the matrix of angle of attack and angle of sideslip
        may be different at each Mach number.  In this case, *db.bkpts*
        will be a list of 1D arrays for *alpha* and *beta* and just a
        single 1D array for *mach*.

        :Call:
            >>> y = db.rcall_multilinear_schedule(col, args, x)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`tuple`
                Values for each argument in *args*
            *tol*: {``1e-6``} | :class:`float` >= 0
                Tolerance for matching slice key
        :Outputs:
            *y*: ``None`` | :class:`float` | ``db[col].__class__``
                Interpolated value from ``db[col]``
        :Versions:
            * 2019-04-19 ``@ddalle``: Version 1.0
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Slice tolerance
        tol = kw.get("tol", 1e-6)
        # Name of master (slice) key
        skey = args[0]
        # Extrapolation option
        extrap = kw.get("extrap", False)
        # Copy arguments
        args = list(args)
        # Get lookup points at both sides of scheduling key
        i0, i1, f, x0, x1 = self.get_schedule(args, x, extrap=extrap)
        # Get the values for the slice key
        x00 = self.get_bkpt(skey, i0)
        x01 = self.get_bkpt(skey, i1)
        # Find indices of the two slices
        I0 = np.where(np.abs(self[skey] - x00) <= tol)[0]
        I1 = np.where(np.abs(self[skey] - x01) <= tol)[0]
        # Perform interpolations
        y0 = self._rcall_multilinear(col, args, x0, I=I0, j=i0)
        y1 = self._rcall_multilinear(col, args, x1, I=I1, j=i1)
        # Linear interpolation in the schedule key
        return (1-f)*y0 + f*y1

   # --- Radial Basis Functions ---
    # RBF lookup
    def rcall_rbf(self, col, args, *x, **kw):
        """Evaluate a single radial basis function

        :Call:
            >>> y = DBc.rcall_rbf(col, args, *x)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`tuple`
                Values for each argument in *args*
        :Outputs:
            *y*: :class:`float` | :class:`np.ndarray`
                Interpolated value from *db[col]*
        :Versions:
            * 2018-12-31 ``@ddalle``: Version 1.0
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Get the radial basis function
        f = self.get_rbf(col)
        # Evaluate
        return f(*x)

    # Get an RBF
    def get_rbf(self, col, *I):
        r"""Extract a radial basis function, with error checking

        :Call:
            >>> f = db.get_rbf(col, *I)
            >>> f = db.get_rbf(col)
            >>> f = db.get_rbf(col, i)
            >>> f = db.get_rbf(col, i, j)
            >>> f = db.get_rbf(col, i, j, ...)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *I*: :class:`tuple`
                Tuple of lookup indices
            *i*: :class:`int`
                (Optional) first RBF list index
            *j*: :class:`int`
                (Optional) second RBF list index
        :Outputs:
            *f*: :class:`scipy.interpolate.rbf.Rbf`
                Callable radial basis function
        :Versions:
            * 2018-12-31 ``@ddalle``: Version 1.0
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Get the radial basis function
        try:
            fn = self.rbf[col]
        except AttributeError:
            # No radial basis functions at all
            raise AttributeError("No radial basis functions found")
        except KeyError:
            # No RBF for this coefficient
            raise KeyError("No radial basis function for col '%s'" % col)
        # Number of indices given
        nd = len(I)
        # Loop through indices
        for n, i in enumerate(I):
            # Try to extract
            try:
                # Get the *ith* list entry
                fn = fn[i]
            except TypeError:
                # Reached RBF too soon
                raise TypeError(
                    ("RBF for '%s':\n" % col) +
                    ("Expecting %i-dimensional " % nd) +
                    ("array but found %i-dim" % n))
        # Test type
        if not callable(fn):
            raise TypeError("RBF '%s' index %i is not callable" % (col, I))
        # Output
        return fn

   # --- RBF-linear ---
    # Multiple RBF lookup
    def rcall_rbf_linear(self, col, args, *x, **kw):
        r"""Evaluate two RBFs at slices of first *arg* and interpolate

        :Call:
            >>> y = db.rcall_rbf_linear(col, args, x)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`tuple`
                Values for each argument in *args*
        :Outputs:
            *y*: :class:`float` | :class:`np.ndarray`
                Interpolated value from *db[col]*
        :Versions:
            * 2018-12-31 ``@ddalle``: Version 1.0
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Lookup value for first variable
        i0, i1, f = self.get_bkpt_index(args[0], x[0])
        # Get lookup functions for *i0* and *i1*
        f0 = self.get_rbf(col, i0)
        f1 = self.get_rbf(col, i1)
        # Evaluate both functions
        y0 = f0(*x[1:])
        y1 = f1(*x[1:])
        # Interpolate
        y = (1-f)*y0 + f*y1
        # Output
        return y

   # --- RBF-schedule ---
    # Multiple RBF lookup, curvilinear
    def rcall_rbf_schedule(self, col, args, *x, **kw):
        r"""Evaluate two RBFs at slices of first *arg* and interpolate

        :Call:
            >>> y = db.rcall_rbf_schedule(col, args, *x)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`tuple`
                Values for each argument in *args*
        :Outputs:
            *y*: :class:`float` | :class:`np.ndarray`
                Interpolated value from *db[col]*
        :Versions:
            * 2018-12-31 ``@ddalle``: Version 1.0
        """
        # Extrapolation option
        extrap = kw.get("extrap", False)
        # Get lookup points at both sides of scheduling key
        i0, i1, f, x0, x1 = self.get_schedule(list(args), x, extrap=extrap)
        # Get lookup functions for *i0* and *i1*
        f0 = self.get_rbf(col, i0)
        f1 = self.get_rbf(col, i1)
        # Evaluate the RBFs at both slices
        y0 = f0(*x0)
        y1 = f1(*x1)
        # Interpolate between the slices
        y = (1-f)*y0 + f*y1
        # Output
        return y

   # --- Generic Function ---
    # Generic function
    def rcall_function(self, col, args, *x, **kw):
        r"""Evaluate a single user-saved function

        :Call:
            >>> y = db.rcall_function(col, args, *x)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to evaluate
            *args*: :class:`list` | :class:`tuple`
                List of lookup key names
            *x*: :class:`tuple`
                Values for each argument in *args*
        :Outputs:
            *y*: ``None`` | :class:`float` | ``DBc[coeff].__class__``
                Interpolated value from ``DBc[coeff]``
        :Versions:
            * 2018-12-31 ``@ddalle``: Version 1.0
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Get the function
        f = self.get_response_func(col)
        # Evaluate
        if self.response_funcs_self.get(col):
            # Use reference to *self*
            return f(self, *x, **kw)
        else:
            # Stand-alone function
            return f(*x, **kw)
  # >

  # ===================
  # Column Names
  # ===================
  # <
   # --- Rename ---
    # Rename a column
    def rename_col(self, col1, col2):
        r"""Rename a column from *col1* to *col2*

        :Call:
            >>> db.rename_col(col1, col2)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col1*: :class:`str`
                Name of column in *db* to rename
            *col2*: :class:`str`
                Renamed title for *col1*
        :Versions:
            * 2021-09-10 ``@ddalle``: Version 1.0
        """
        # Check if *col1* is present
        if col1 not in self:
            return
        # Get index
        if col1 in self.cols:
            # Find it
            i = self.cols.index(col1)
            # replace it
            self.cols[i] = col2
        # Get definition
        if col1 in self.defns:
            # Remove it
            defn = self.defns.pop(col1)
            # Save it
            self.set_defn(col2, defn)
        # Save new column
        self.save_col(col2, self.pop(col1))
            
   # --- Prefix ---
    # Prepend something to the name of a column
    def prepend_colname(self, col, prefix):
        r"""Add a prefix to a column name

        This maintains component names, so for example if *col* is
        ``"bullet.CN"``, and *prefix* is ``"U"``, the result is
        ``"bullet.UCN"``.

        :Call:
            >>> newcol = db.prepend_colname(col, prefix)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to prepend
            *prefix*: :class:`str`
                Prefix to prefix
        :Outputs:
            *newcol*: :class:`str`
                Prefixed name
        :Versions:
            * 2020-03-24 ``@ddalle``: Version 1.0
        """
        # Check for null input
        if not prefix:
            return col
        # Check for component name
        parts = col.split(".")
        # Component name
        comp = ".".join(parts[:-1])
        # Original critical part
        coeff = parts[-1]
        # Reassemble parts
        if comp:
            # Preserve component name
            newcol = comp + "." + prefix + coeff
        else:
            # No component name
            newcol = prefix + coeff
        # Output
        return newcol

    # Remove prefix from the name of a column
    def lstrip_colname(self, col, prefix):
        r"""Remove a prefix from a column name

        This maintains component names, so for example if *col* is
        ``"bullet.UCN"``, and *prefix* is ``"U"``, the result is
        ``"bullet.CN"``.

        :Call:
            >>> newcol = db.lstrip_colname(col, prefix)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to strip
            *prefix*: :class:`str`
                Prefix to remove
        :Outputs:
            *newcol*: :class:`str`
                Prefixed name
        :Versions:
            * 2020-03-24 ``@ddalle``: Version 1.0
        """
        # Check for null input
        if not prefix:
            return col
        # Check for component name
        parts = col.split(".")
        # Component name
        comp = ".".join(parts[:-1])
        # Original critical part
        coeff = parts[-1]
        # Remove prefix
        if coeff.startswith(prefix):
            # Strip it
            coeff = coeff[len(prefix):]
        # Reassemble parts
        if comp:
            # Preserve component name
            newcol = comp + "." + coeff
        else:
            # No component name
            newcol = coeff
        # Output
        return newcol

    # Substitute suffix
    def substitute_prefix(self, col, prefix1, prefix2):
        r"""Remove a prefix from a column name

        This maintains component names, so for example if *col* is
        ``"bullet.CLMF"``, *prefix1* is ``"CLM"``, *suffix2* is
        ``"CN"``, and the result is ``"bullet.CNF"``.

        :Call:
            >>> newcol = db.substitute_prefix(col, prefix1, prefix2)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to strip
            *prefix1*: :class:`str`
                Prefix to remove from column name
            *prefix2*: :class:`str`
                Prefix to add to column name
        :Outputs:
            *newcol*: :class:`str`
                Prefixed name
        :Versions:
            * 2020-03-24 ``@ddalle``: Version 1.0
        """
        # Check for component name
        parts = col.split(".")
        # Component name
        comp = ".".join(parts[:-1])
        # Original critical part
        coeff = parts[-1]
        # Remove prefix
        if coeff.startswith(prefix1):
            # Replace it
            coeff = prefix2 + coeff[len(prefix1):]
        else:
            # Add prefix anyway
            coeff = prefix2 + coeff
        # Reassemble parts
        if comp:
            # Preserve component name
            newcol = comp + "." + coeff
        else:
            # No component name
            newcol = coeff
        # Output
        return newcol

   # --- Suffix ---
    # Append something to the name of a column
    def append_colname(self, col, suffix):
        r"""Add a suffix to a column name

        This maintains component names, so for example if *col* is
        ``"bullet.CLM"``, and *suffix* is ``"X"``, the result is
        ``"bullet.CLMX"``.

        :Call:
            >>> newcol = db.append_colname(col, suffix)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to append
            *suffix*: :class:`str`
                Suffix to append to column name
        :Outputs:
            *newcol*: :class:`str`
                Prefixed name
        :Versions:
            * 2020-03-24 ``@ddalle``: Version 1.0
        """
        # Check for null input
        if not suffix:
            return col
        # Check for component name
        parts = col.split(".")
        # Component name
        comp = ".".join(parts[:-1])
        # Original critical part
        coeff = parts[-1]
        # Reassemble parts
        if comp:
            # Preserve component name
            newcol = comp + "." + coeff + suffix
        else:
            # No component name
            newcol = coeff + suffix
        # Output
        return newcol

    # Prepend something to the name of a columns
    def rstrip_colname(self, col, suffix):
        r"""Remove a suffix from a column name

        This maintains component names, so for example if *col* is
        ``"bullet.CLMX"``, and *suffix* is ``"X"``, the result is
        ``"bullet.CLM"``.

        :Call:
            >>> newcol = db.rstrip_colname(col, suffix)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to strip
            *suffix*: :class:`str`
                Suffix to remove from column name
        :Outputs:
            *newcol*: :class:`str`
                Prefixed name
        :Versions:
            * 2020-03-24 ``@ddalle``: Version 1.0
        """
        # Check for null input
        if not suffix:
            return col
        # Check for component name
        parts = col.split(".")
        # Component name
        comp = ".".join(parts[:-1])
        # Original critical part
        coeff = parts[-1]
        # Remove prefix
        if coeff.endswith(suffix):
            # Strip it
            coeff = coeff[:-len(suffix)]
        # Reassemble parts
        if comp:
            # Preserve component name
            newcol = comp + "." + coeff
        else:
            # No component name
            newcol = coeff
        # Output
        return newcol

    # Substitute suffix
    def substitute_suffix(self, col, suffix1, suffix2):
        r"""Remove a suffix from a column name

        This maintains component names, so for example if *col* is
        ``"bullet.CLM"``, *suffix1* is ``"LM"``, *suffix2* is ``"N"``,
        and the result is ``"bullet.CN"``.

        :Call:
            >>> newcol = db.substitute_suffix(col, suffix1, suffix2)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to strip
            *suffix1*: :class:`str`
                Suffix to remove from column name
            *suffix2*: :class:`str`
                Suffix to add to column name
        :Outputs:
            *newcol*: :class:`str`
                Prefixed name
        :Versions:
            * 2020-03-24 ``@ddalle``: Version 1.0
        """
        # Check for component name
        parts = col.split(".")
        # Component name
        comp = ".".join(parts[:-1])
        # Original critical part
        coeff = parts[-1]
        # Remove prefix
        if coeff.endswith(suffix1):
            # Replace it
            coeff = coeff[:-len(suffix1)] + suffix2
        else:
            # Add suffix anyway
            coeff = coeff + suffix2
        # Reassemble parts
        if comp:
            # Preserve component name
            newcol = comp + "." + coeff
        else:
            # No component name
            newcol = coeff
        # Output
        return newcol
  # >

  # ===================
  # UQ
  # ===================
  # <
   # --- Estimators ---
    # Entire database UQ generation
    def est_uq_db(self, db2, cols=None, **kw):
        r"""Quantify uncertainty for all *col*, *ucol* pairings in DB

        :Call:
            >>> db1.est_uq_db(db2, cols=None, **kw)
        :Inputs:
            *db1*: :class:`DataKit`
                Database with scalar output functions
            *db2*: :class:`DataKit`
                Target database (UQ based on difference)
            *cols*: {``None``} | :class:`list`\ [:class:`str`]
                Data columns to estimate UQ (default is all *db1.cols*
                that have a *ucol* defined)
        :Keyword Arguments:
            *nmin*: {``30``} | :class:`int` > 0
                Minimum number of points in window
            *cov*, *Coverage*: {``0.99865``} | 0 < :class:`float` < 1
                Fraction of data that must be covered by UQ term
            *cdf*, *CoverageCDF*: {*cov*} | 0 < :class:`float` < 1
                Coverage fraction assuming perfect distribution
            *test_values*: {``{}``} | :class:`dict`
                Candidate values of each *col* for comparison
            *test_bkpts*: {``{}``} | :class:`dict`
                Candidate break points (1D unique) for *col*
        :Required Attributes:
            *db1.uq_cols*: :class:`dict`\ [:class:`list`]
                Names of UQ col for each *col*, if any
            *db1.response_args[col]*: :class:`list`\ [:class:`str`]
                List of args to evaluate *col*
            *db1.response_args[ucol]*: :class:`list`\ [:class:`str`]
                List of args to evaluate *ucol*
            *db1.uq_ecols[ucol]*: {``[]``} | :class:`list`
                List of extra UQ cols related to *ucol*
            *db1.uq_acols[ucol]*: {``[]``} | :class:`list`
                Aux cols whose deltas are used to estimate *ucol*
            *db1.uq_efuncs*: {``{}``} | :class:`dict`\ [**callable**]
                Function to calculate any *uq_ecols*
            *db1.uq_afuncs*: {``{}``} | :class:`dict`\ [**callable**]
                Function to use aux cols when estimating *ucol*
        :Versions:
            * 2019-02-15 ``@ddalle``: Version 1.0
            * 2020-04-02 ``@ddalle``: Version 2.0
                - was :func:`EstimateUQ_DB`
        """
       # --- Inputs ---
        # Get minimum number of points in statistical window
        nmin = kw.get("nmin", 30)
        # Get columns
        if cols is None:
            # Initialize (use any col with a *ucol*)
            cols = []
            # Loop through columns
            for col in self.cols:
                # Get UQ col
                ucols = self.get_uq_col(col)
                # Include this *col* if a UQ col is defined
                if ucols:
                    cols.append(col)
       # --- Column Loop ---
        # Loop through data coefficients
        for col in cols:
            # Get UQ col list
            ucols = self.get_uq_col(col)
            # Skip if no UQ cols
            if not ucols:
                continue
            # Enlist
            if typeutils.isstr(ucols):
                # Make single list
                ucols = [ucols]
            # Loop through them
            for ucol in ucols:
                # Status update
                sys.stdout.write("%-60s\r" %
                    ("Estimating UQ: %s --> %s" % (col, ucol)))
                sys.stdout.flush()
                # Process "extra" keys
                uq_ecols = self.get_uq_ecol(ucol)
                # Call particular method
                A, U = self.est_uq_col(db2, col, ucol, **kw)
                # Save primary key
                self.save_col(ucol, U[:,0])
                # Save additional keys
                for (j, acol) in enumerate(uq_ecols):
                    # Save additional key values
                    self.save_col(acol, U[:,j+1])
        # Clean up prompt
        sys.stdout.write("%60s\r" % "")

    # UQ estimates for each condition in a col
    def est_uq_col(self, db2, col, ucol, **kw):
        r"""Quantify uncertainty interval for all points of one *ucol*

        :Call:
            >>> A, U  = db1.est_uq_col(db2, col, ucol, **kw)
        :Inputs:
            *db1*: :class:`DataKit`
                Database with scalar output functions
            *db2*: :class:`DataKit`
                Target database (UQ based on difference)
            *col*: :class:`str`
                Name of data column to analyze
            *ucol*: :class:`str`
                Name of UQ column to estimate
        :Keyword Arguments:
            *nmin*: {``30``} | :class:`int` > 0
                Minimum number of points in window
            *cov*, *Coverage*: {``0.99865``} | 0 < :class:`float` < 1
                Fraction of data that must be covered by UQ term
            *cdf*, *CoverageCDF*: {*cov*} | 0 < :class:`float` < 1
                Coverage fraction assuming perfect distribution
            *test_values*: {``{}``} | :class:`dict`
                Candidate values of each *response_arg* for comparison
            *test_bkpts*: {``{}``} | :class:`dict`
                Candidate break points (1D unique) for *response_args*
        :Required Attributes:
            *db1.response_args[col]*: :class:`list`\ [:class:`str`]
                List of args to evaluate *col*
            *db1.response_args[ucol]*: :class:`list`\ [:class:`str`]
                List of args to evaluate *ucol*
            *db1.uq_ecols[ucol]*: {``[]``} | :class:`list`
                List of extra UQ cols related to *ucol*
            *db1.uq_acols[ucol]*: {``[]``} | :class:`list`
                Aux cols whose deltas are used to estimate *ucol*
            *db1.uq_efuncs*: {``{}``} | :class:`dict`\ [**callable**]
                Function to calculate any *uq_ecols*
            *db1.uq_afuncs*: {``{}``} | :class:`dict`\ [**callable**]
                Function to use aux cols when estimating *ucol*
        :Outputs:
            *A*: :class:`np.ndarray` size=(*nx*\ ,*na*\ )
                Conditions for each *ucol* window, for *nx* windows,
                each with *na* values (length of
                *db1.response_args[ucol]*)
            *U*: :class:`np.ndarray` size=(*nx*\ ,*nu*\ +1)
                Values of *ucol* and any *nu* "extra" *uq_ecols* for
                each window
        :Versions:
            * 2019-02-15 ``@ddalle``: Version 1.0
            * 2020-04-02 ``@ddalle``: v2.0, from ``EstimateUQ_coeff()``
        """
       # --- Inputs ---
        # Get eval arguments for input coeff and UQ coeff
        uargs = self.get_response_args(ucol)
        # Get minimum number of points in statistical window
        nmin = kw.get("nmin", 30)
        # Additional information
        uq_ecols = self.get_uq_ecol(ucol)
        # Check length
        if uq_ecols is None:
            # No extra ecols
            nu = 1
        else:
            # Total number of cols
            nu = 1 + len(uq_ecols)
       # --- Windowing ---
        # Break up possible run matrix into window centers
        A = self._genr8_uq_conditions(uargs, **kw)
        # Number of windows
        nx = len(A)
        # Initialize output
        U = np.zeros((nx, nu))
       # --- Evaluation ---
        # Loop through conditions
        for (i, a) in enumerate(A):
            # Get window
            I = self.genr8_window(nmin, uargs, *a, **kw)
            # Estimate UQ for this window
            U[i] = self._est_uq_point(db2, col, ucol, I, **kw)
       # --- Output ---
        # Return conditions and values
        return A, U

    # Estimate UQ at a single *ucol* condition
    def est_uq_point(self, db2, col, ucol, *a, **kw):
        r"""Quantify uncertainty interval for a single point or window

        :Call:
            >>> u, U = db1.est_uq_point(db2, col, ucol, *a, **kw)
        :Inputs:
            *db1*: :class:`DataKit`
                Database with scalar output functions
            *db2*: :class:`DataKit`
                Target database (UQ based on difference)
            *col*: :class:`str`
                Name of data column to analyze
            *ucol*: :class:`str`
                Name of UQ column to estimate
            *a*: :class:`tuple`\ [:class:`float`]
                Conditions at which to evaluate uncertainty
            *a[0]*: :class:`float`
                Value of *db1.response_args[ucol][0]*
        :Keyword Arguments:
            *nmin*: {``30``} | :class:`int` > 0
                Minimum number of points in window
            *cov*, *Coverage*: {``0.99865``} | 0 < :class:`float` < 1
                Fraction of data that must be covered by UQ term
            *cdf*, *CoverageCDF*: {*cov*} | 0 < :class:`float` < 1
                Coverage fraction assuming perfect distribution
            *test_values*: {``{}``} | :class:`dict`
                Candidate values of each *response_args* for comparison
            *test_bkpts*: {``{}``} | :class:`dict`
                Candidate break points (1D unique) for *response_args*
        :Required Attributes:
            *db1.response_args[col]*: :class:`list`\ [:class:`str`]
                List of args to evaluate *col*
            *db1.response_args[ucol]*: :class:`list`\ [:class:`str`]
                List of args to evaluate *ucol*
            *db1.uq_ecols[ucol]*: {``[]``} | :class:`list`
                List of extra UQ cols related to *ucol*
            *db1.uq_acols[ucol]*: {``[]``} | :class:`list`
                Aux cols whose deltas are used to estimate *ucol*
            *db1.uq_efuncs*: {``{}``} | :class:`dict`\ [**callable**]
                Function to calculate any *uq_ecols*
            *db1.uq_afuncs*: {``{}``} | :class:`dict`\ [**callable**]
                Function to use aux cols when estimating *ucol*
        :Outputs:
            *u*: :class:`float`
                Single uncertainty estimate for generated window
            *U*: :class:`tuple`\ [:class:`float`]
                Values of any "extra" *uq_ecols*
        :Versions:
            * 2019-02-15 ``@ddalle``: Version 1.0
            * 2020-04-02 ``@ddalle``: Second version
        """
       # --- Inputs ---
        # Get eval arguments for input coeff and UQ coeff
        uargs = self.get_response_args(ucol)
        # Get minimum number of points in statistical window
        nmin = kw.get("nmin", 30)
       # --- Windowing ---
        # Get window
        I = self.genr8_window(nmin, uargs, *a, **kw)
       # --- Evaluation ---
        # Call stand-alone method
        U = self._est_uq_point(db2, col, ucol, I, **kw)
       # --- Output ---
        # Return all values
        return U

    # Estimate UQ at one eval point of *ucol*
    def _est_uq_point(self, db2, col, ucol, mask, **kw):
        r"""Quantify uncertainty interval for a single point or window

        :Call:
            >>> u, a = db1.est_uq_point(db2, col, ucol, mask, **kw)
            >>> u, a = db1.est_uq_point(db2, col, I, mask, **kw)
        :Inputs:
            *db1*: :class:`DataKit`
                Database with scalar output functions
            *db2*: :class:`DataKit`
                Target database (UQ based on difference)
            *col*: :class:`str`
                Name of data column to analyze
            *ucol*: :class:`str`
                Name of UQ column to estimate
            *mask*: :class:`np.ndarray`\ [:class:`bool`]
                Mask of which *db1[col]* indices to consider
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Indices of *db1[col]* to consider
        :Required Attributes:
            *db1.response_args[col]*: :class:`list`\ [:class:`str`]
                List of args to evaluate *col*
            *db1.response_args[ucol]*: :class:`list`\ [:class:`str`]
                List of args to evaluate *ucol*
            *db1.uq_ecols[ucol]*: {``[]``} | :class:`list`
                List of extra UQ cols related to *ucol*
            *db1.uq_acols[ucol]*: {``[]``} | :class:`list`
                Aux cols whose deltas are used to estimate *ucol*
            *db1.uq_efuncs*: {``{}``} | :class:`dict`\ [**callable**]
                Function to calculate any *uq_ecols*
            *db1.uq_afuncs*: {``{}``} | :class:`dict`\ [**callable**]
                Function to use aux cols when estimating *ucol*
        :Outputs:
            *u*: :class:`float`
                Single uncertainty estimate for generated window
            *a*: :class:`tuple`\ [:class:`float`]
                Values of any "extra" *uq_ecols*
        :Versions:
            * 2019-02-15 ``@ddalle``: Version 1.0
            * 2020-03-20 ``@ddalle``: Mods from :mod:`tnakit.db.db1`
        """
       # --- Statistics Options ---
        # Probability
        cov = kw.get("Coverage", kw.get("cov", 0.99865))
        cdf = kw.get("CoverageCDF", kw.get("cdf", cov))
        # Outlier cutoff
        osig_kw = kw.get('OutlierSigma', kw.get("osig"))
       # --- Attributes ---
        # Unpack useful attributes for additional functions
        uq_funcs_extra = getattr(self, "uq_funcs_extra", {})
        uq_funcs_shift = getattr(self, "uq_funcs_shift", {})
        # Additional information
        uq_keys_extra = getattr(self, "uq_keys_extra", {})
        uq_keys_shift = getattr(self, "uq_keys_shift", {})
        # Get eval arguments for input coeff and UQ coeff
        argsc = self.response_args[col]
        argsu = self.response_args[ucol]
       # --- Test Conditions ---
        # Get test values and test break points
        vals, bkpts = self._get_test_values(argsc, **kw)
        # --- Evaluation ---
        # Initialize input values for comparison evaluation
        A = []
        # Get dictionary values
        for k in argsc:
            # Apply mask
            A.append(vals[k][mask])
        # Evaluate both databases
        V1 = self(col, *A)
        V2 = db2(col, *A)
        # Deltas
        DV = V2 - V1
       # --- Aux cols ---
        # Get aux cols require to estimate *ucol*
        acols = self.get_uq_acol(ucol)
        # Deltas of co-keys
        DV0_aux = []
        # Loop through shift keys
        for acol in acols:
            # Evaluate both databases
            V1 = self(acol, *A)
            V2 = db2(acol, *A)
            # Append deltas
            DV0_aux.append(V2-V1)
       # --- Outliers ---
        # Degrees of freedom
        df = DV.size
        # Nominal bounds (like 3-sigma for 99.5% coverage, etc.)
        ksig = statutils.student.ppf(0.5+0.5*cdf, df)
        kcov = statutils.student.ppf(0.5+0.5*cov, df)
        # Outlier cutoff
        if osig_kw is None:
            # Default
            osig = 1.5*ksig
        else:
            # User-supplied value
            osig = osig_kw
        # Check outliers on main deltas
        J = statutils.check_outliers(DV, cov, cdf=cdf, osig=osig)
        # Loop through shift keys; points must be non-outlier in all keys
        for DVk in DV0_aux:
            # Check outliers in these deltas
            Jk = statutils.check_outliers(DVk, cov, cdf=cdf, osig=osig)
            # Combine constraints
            J = np.logical_and(J, Jk)
        # Downselect original deltas
        DV = DV[J]
        # Initialize downselected correlated deltas
        DV_aux = []
        # Downselect correlated deltas
        for DVk in DV0_aux:
            DV_aux.append(DVk[J])
        # New degrees of freedom
        df = DV.size
        # Nominal bounds (like 3-sigma for 99.5% coverage, etc.)
        ksig = statutils.student.ppf(0.5+0.5*cdf, df)
        kcov = statutils.student.ppf(0.5+0.5*cov, df)
        # Outlier cutoff
        if osig_kw is None:
            # Default
            osig = 1.5*ksig
        else:
            # User-supplied value
            osig = osig_kw
       # --- Extra cols ---
        # List of extra keys
        ecols = self.get_uq_ecol(ucol)
        # Initialize tuple of extra key values
        a_extra = []
        # Loop through extra keys
        for ecol in ecols:
            # Get function
            fn = self.get_uq_efunc(ecol)
            # This function is required
            if fn is None:
                raise ValueError("No function for extra UQ col '%s'" % ecol)
            # Evaluate
            a_extra.append(fn(self, DV, *DV_aux))
       # --- Delta shifting (aux functions) ---
        # Function to perform any shifts
        afunc = self.get_uq_afunc(ucol)
        # Perform shift
        if afunc is not None:
            # Extra arguments for shift key
            a_aux = DV_aux + a_extra
            # Perform shift using appropriate function
            DV = afunc(self, DV, *a_aux)
       # --- Statistics ---
        # Calculate coverage interval
        vmin, vmax = statutils.get_cov_interval(DV, cov, cdf=cdf, osig=osig)
        # Max value
        u = max(abs(vmin), abs(vmax))
       # --- Output ---
        # Return all extra values
        return (u,) + tuple(a_extra)

   # --- Grouping ---
    # Find *N* neighbors based on list of args
    def genr8_window(self, n, args, *a, **kw):
        r"""Get indices of neighboring points

        This function creates a moving "window" for averaging or for
        performing other statistics (especially estimating difference
        between two databases).

        :Call:
            >>> I = db.genr8_window(n, args, *a, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with evaluation tools
            *n*: :class:`int`
                Minimum number of points in window
            *args*: :class:`list`\ [:class:`str`]
                List of arguments to use for windowing
            *a[0]*: :class:`float`
                Value of the first argument
            *a[1]*: :class:`float`
                Value of the second argument
        :Keyword Arguments:
            *test_values*: {*db*} | :class:`DBCoeff` | :class:`dict`
                Specify values of each *arg* in *args* that are the
                candidate points for the window; default is from *db*
            *test_bkpts*: {*db.bkpts*} | :class:`dict`
                Specify candidate window boundaries; must be ascending
                array of unique values for each *arg*
        :Outputs:
            *I*: :class:`np.ndarray`
                Indices of cases (relative to *test_values*) in window
        :Versions:
            * 2019-02-13 ``@ddalle``: Version 1.0
            * 2020-04-01 ``@ddalle``: Modified from :mod:`tnakit.db`
        """
       # --- Check bounds ---
        def check_bounds(vmin, vmax, vals):
            # Loop through parameters
            for i,k in enumerate(args):
                # Get values
                Vk = vals[k]
                # Check bounds
                J = np.logical_and(Vk>=vmin[k], Vk<=vmax[k])
                # Combine constraints if i>0
                if i == 0:
                    # First run
                    I = J
                else:
                    # Combine constraints
                    I = np.logical_and(I, J)
            # Output
            return I
       # --- Init ---
        # Check inputs
        if not isinstance(args, (list, tuple)):
            raise TypeError("Arg list must be 'list' (got '%s')" % type(args))
        # Number of args
        narg = len(args)
        # Check inputs
        if narg == 0:
            # No windowing arguments
            raise ValueError("At least one named argument required")
        elif len(a) != narg:
            # Not enough values
            raise ValueError("%i argument names provided but %i values"
                % (narg, len(a)))
        # Get the length of the database for the first argument
        nx = self[args[0]].size
        # Initialize mask
        I = np.arange(nx) > -1
       # --- Lookup values ---
        # Get test values and test break points
        vals, bkpts = self._get_test_values(args, **kw)
       # --- Tolerances/Bounds ---
        # Default tolerance
        tol = kw.get("tol", 1e-8)
        # Tolerance dictionary
        tols = {}
        # Initialize bounds
        vmin = {}
        vmax = {}
        # Loop through parameters
        for i, k in enumerate(args):
            # Get value
            v = a[i]
            # Get tolerance
            tolk = kw.get("%stol" % k, tol)
            # Save it
            tols[k] = tolk
            # Bounds
            vmin[k] = v - tolk
            vmax[k] = v + tolk
       # --- Initial Check ---
        I = check_bounds(vmin, vmax, vals)
        # Initial window count
        m = np.count_nonzero(I)
       # --- Expansion ---
        # Maximum loop
        maxloops = kw.get("nmax", kw.get("maxloops", 10))
        # Loop until enough points are included
        for nloop in range(maxloops):
            # Check count
            if m >= n:
                break
            # Expand each argument in order
            for i, k in enumerate(args):
                # Current bounds
                ak = vmin[k]
                bk = vmax[k]
                # Tolerance
                tolk = tols[k]
                # Break points
                Xk = bkpts[k]
                # Look for next point outside bound
                ja = Xk < ak - tolk
                jb = Xk > bk + tolk
                # Expand bounds if possible
                if np.any(ja):
                    vmin[k] = np.max(Xk[ja])
                if np.any(jb):
                    vmax[k] = np.min(Xk[jb])
                # Check new window
                I = check_bounds(vmin, vmax, vals)
                # Update count
                m = np.count_nonzero(I)
                # Check count
                if m >= n:
                    break
       # --- Output ---
        # Output
        return np.where(I)[0]

    # Get dictionary of test values
    def _get_test_values(self, args, **kw):
        r"""Get test values for creating windows or comparing databases

        :Call:
            >>> vals, bkpts = db._get_test_values(args, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *args*: :class:`list`\ [:class:`str`]
                List of arguments to use for windowing
        :Keyword Arguments:
            *test_values*: {*db*} | :class:`DataKit` | :class:`dict`
                Specify values of each parameter in *args* that are the
                candidate points for the window; default is from *db*
            *test_bkpts*: {*db.bkpts*} | :class:`dict`
                Specify candidate window boundaries; must be ascending
                array of unique values for each key
        :Outputs:
            *vals*: :class:`dict`\ [:class:`np.ndarray`]
                Dictionary of lookup values for each key in *args*
            *bkpts*: :class:`dict`\ [:class:`np.ndarray`]
                Dictionary of unique candidate values for each key
        :Versions:
            * 2019-02-13 ``@ddalle``: Version 1.0
            * 2020-03-20 ``@ddalle``: Migrated from :mod:`tnakit`
        """
       # --- Lookup values ---
        # Values to use (default is direct from database)
        vals = {}
        # Dictionary from keyword args
        test_values = kw.get("test_values", {})
        # Loop through parameters
        for k in args:
            # Get reference values for parameter *k*
            vals[k] = test_values.get(k, self.get_all_values(k))
       # --- Break points ---
        # Values to use for searching
        bkpts = {}
        # Primary default: *DBc.bkpts*
        self_bkpts = self.__dict__.get("bkpts", {})
        # Secondary default: *bkpts* attribute from secondary database
        test_bkpts = test_values.get("test_bkpts", {})
        # User-specified option
        test_bkpts = kw.get("test_bkpts", test_bkpts)
        # Primary default: *bkpts* from this database
        for k in args:
            # Check for specified values
            if k in test_bkpts:
                # Use user-specified or secondary-db values
                bkpts[k] = test_bkpts[k].copy()
            elif k in self_bkpts:
                # Use break points from this database
                bkpts[k] = self_bkpts[k].copy()
            else:
                # Use unique test values
                bkpts[k] = np.unique(vals[k])
       # --- Output ---
        # Return actual values and break points (unique vals)
        return vals, bkpts

    # Find *N* neighbors based on list of args
    def _genr8_uq_conditions(self, uargs, **kw):
        r"""Get list of points at which to estimate UQ database

        :Call:
            >>> A = db._genr8_uq_conditions(uargs, **kw)
            >>> [a0, a1, ...] = db._genr8_uq_conditions(uargs, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *uargs*: :class:`list`\ [:class:`str`]
                List of args to *ucol* for window generation
        :Keyword Arguments:
            *test_values*: {``{}``} | :class:`dict`
                Candidate values of each *uarg* for comparison
            *test_bkpts*: {``{}``} | :class:`dict`
                Candidate break points (1D unique) for *uargs*
        :Outputs:
            *A*: :class:`np.ndarray`\ [:class:`float`]
                List of conditions for each window
            *a0*: :class:`np.ndarray`\ [:class:`float`]
                Value of *uargs* for first window
            *a01*: :class:`float`
                Value of *uargs[1]* for second window
        :Versions:
            * 2019-02-16 ``@ddalle``: Version 1.0
            * 2020-04-02 ``@ddalle``: Updates for :class:`DataKit`
        """
        # Get test values and test break points
        vals, bkpts = self._get_test_values(uargs, **kw)
        # Number of args
        narg = len(uargs)
        # Create tuple of all breakpoints
        bkpts1d = tuple(bkpts[k] for k in uargs)
        # Create *n*-dimensional array of points (full factorial)
        vals_nd = np.meshgrid(*bkpts1d, indexing="ij")
        # Flatten each entry and make into a row vector
        V = tuple([v.flatten()] for v in vals_nd)
        # Combine into single vector
        A = np.vstack(V).T
        # Output
        return A

   # --- Options: Get ---
    # Get UQ coefficient
    def get_uq_col(self, col):
        r"""Get name of UQ columns for *col*

        :Call:
            >>> ucol = db.get_uq_col(col)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column to evaluate
        :Outputs:
            *ucol*: ``None`` | :class:`str`
                Name of UQ column for *col*
        :Versions:
            * 2019-03-13 ``@ddalle``: Version 1.0
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
            * 2019-12-26 ``@ddalle``: Renamed from :func:`get_uq_coeff`
        """
        # Get dictionary of UQ cols
        uq_cols = self.__dict__.setdefault("uq_cols", {})
        # Get entry for this coefficient
        return uq_cols.get(col)

    # Get extra UQ cols
    def get_uq_ecol(self, ucol):
        r"""Get names of any extra UQ cols related to primary UQ col

        :Call:
            >>> ecols = db.get_uq_ecol(ucol)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *ucol*: :class:`str`
                Name of UQ column to evaluate
        :Outputs:
            *ecols*: :class:`list`\ [:class:`str`]
                Name of extra columns required to evaluate *ucol*
        :Versions:
            * 2020-03-21 ``@ddalle``: Version 1.0
        """
        # Get dictionary of ecols
        uq_ecols = self.__dict__.get("uq_ecols", {})
        # Get ecols
        ecols = uq_ecols.get(ucol, [])
        # Check type
        if typeutils.isstr(ecols):
            # Create single list
            ecols = [ecols]
        elif ecols is None:
            # Empty result should be empty list
            ecols = []
        elif not isinstance(ecols, list):
            # Invalid type
            raise TypeError(
                "uq_ecols for col '%s' should be list; got '%s'"
                % (ucol, type(ecols)))
        # Return it
        return ecols

    # Get extra functions for uq
    def get_uq_efunc(self, ecol):
        r"""Get function to evaluate extra UQ column

        :Call:
            >>> efunc = db.get_uq_efunc(ecol)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *ecol*: :class:`str`
                Name of (correlated) UQ column to evaluate
        :Outputs:
            *efunc*: **callable**
                Function to evaluate *ecol*
        :Versions:
            * 2020-03-20 ``@ddalle``: Version 1.0
        """
        # Get dictionary of extra UQ funcs
        uq_efuncs = self.__dict__.get("uq_efuncs", {})
        # Get entry for *col*
        return uq_efuncs.get(ecol)

    # Get aux columns needed to compute UQ of a col
    def get_uq_acol(self, ucol):
        r"""Get name of aux data cols needed to compute UQ col

        :Call:
            >>> acols = db.get_uq_acol(ucol)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *ucol*: :class:`str`
                Name of UQ column to evaluate
        :Outputs:
            *acols*: :class:`list`\ [:class:`str`]
                Name of extra columns required for estimate *ucol*
        :Versions:
            * 2020-03-23 ``@ddalle``: Version 1.0
        """
        # Get dictionary of acols
        uq_acols = self.__dict__.get("uq_acols", {})
        # Get acols
        acols = uq_acols.get(ucol, [])
        # Check type
        if typeutils.isstr(acols):
            # Create single list
            acols = [acols]
        elif acols is None:
            # Empty result should be empty list
            acols = []
        elif not isinstance(acols, list):
            # Invalid type
            raise TypeError(
                "uq_acols for col '%s' should be list; got '%s'"
                % (ucol, type(acols)))
        # Return it
        return acols

    # Get functions to compute UQ for aux col
    def get_uq_afunc(self, ucol):
        r"""Get function to UQ column if aux cols are present

        :Call:
            >>> afunc = db.get_uq_afunc(ucol)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *ucol*: :class:`str`
                Name of UQ col to estimate
        :Outputs:
            *afunc*: **callable**
                Function to estimate *ucol*
        :Versions:
            * 2020-03-23 ``@ddalle``: Version 1.0
        """
        # Get dictionary of aux UQ funcs
        uq_afuncs = self.__dict__.get("uq_afuncs", {})
        # Get entry for *col*
        return uq_afuncs.get(ucol)

   # --- Options: Set ---
    # Set name of UQ col for given col
    def set_uq_col(self, col, ucol):
        r"""Set uncertainty column name for given *col*

        :Call:
            >>> db.set_uq_col(col, ucol)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column
            *ucol*: ``None`` | :class:`str`
                Name of column for UQ of *col* (remove if ``None``)
        :Effects:
            *db.uq_cols*: :class:`dict`
                Entry for *col* set to *ucol*
        :Versions:
            * 2020-03-20 ``@ddalle``: Version 1.0
            * 2020-05-08 ``@ddalle``: Remove if *ucol* is ``None``
        """
        # Get handle to attribute
        uq_cols = self.__dict__.setdefault("uq_cols", {})
        # Check type
        if not isinstance(uq_cols, dict):
            raise TypeError("uq_cols attribute is not a dict")
        # Check trivial
        if ucol is None:
            # Remove it
            uq_cols.pop(col, None)
            return
        # Check types
        if not typeutils.isstr(col):
            raise TypeError(
                "Data column name must be str (got %s)" % type(col))
        if not typeutils.isstr(ucol):
            raise TypeError(
                "UQ column name must be str (got %s)" % type(ucol))
        # Set parameter
        uq_cols[col] = ucol

    # Get extra UQ cols
    def set_uq_ecol(self, ucol, ecols):
        r"""Get name of any extra cols required for a UQ col

        :Call:
            >>> db.get_uq_ecol(ucol, ecol)
            >>> db.get_uq_ecol(ucol, ecols)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *ucol*: :class:`str`
                Name of UQ column to evaluate
            *ecol*: :class:`str`
                Name of extra column required for *ucol*
            *ecols*: :class:`list`\ [:class:`str`]
                Name of extra columns required for *ucol*
        :Versions:
            * 2020-03-21 ``@ddalle``: Version 1.0
            * 2020-05-08 ``@ddalle``: Remove if *ecols* is ``None``
        """
        # Get dictionary of ecols
        uq_ecols = self.__dict__.setdefault("uq_ecols", {})
        # Check for trivial case
        if ecols is None:
            # Remove it
            uq_ecols.pop(ucol, None)
            return
        # Check type
        if typeutils.isstr(ecols):
            # Create single list
            ecols = [ecols]
        elif ecols is None:
            # Empty result should be empty list
            ecols = []
        elif not isinstance(ecols, list):
            # Invalid type
            raise TypeError(
                "uq_ecols for col '%s' should be list; got '%s'"
                % (ucol, type(ecols)))
        # Set it
        uq_ecols[ucol] = ecols

    # Set extra functions for uq
    def set_uq_efunc(self, ecol, efunc):
        r"""Set function to evaluate extra UQ column

        :Call:
            >>> db.set_uq_ecol_funcs(ecol, efunc)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *ecol*: :class:`str`
                Name of (correlated) UQ column to evaluate
            *efunc*: ``None`` | **callable**
                Function to evaluate *ecol*
        :Versions:
            * 2020-03-21 ``@ddalle``: Version 1.0
            * 2020-05-08 ``@ddalle``: Remove if *efunc* is ``None``
        """
        # Get dictionary of extra UQ funcs
        uq_efuncs = self.__dict__.setdefault("uq_efuncs", {})
        # Check input
        if efunc is None:
            # Remove it
            uq_efuncs.pop(ecol, None)
            return
        elif not callable(efunc):
            # Bad function
            raise TypeError("Function is not callable")
        # Get entry for *col*
        uq_efuncs[ecol] = efunc

    # Get aux columns needed to compute UQ of a col
    def set_uq_acol(self, ucol, acols):
        r"""Set name of extra data cols needed to compute UQ col

        :Call:
            >>> db.set_uq_acol(ucol, acols)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *ucol*: :class:`str`
                Name of UQ column to evaluate
            *acols*: ``None`` | :class:`list`\ [:class:`str`]
                Name of extra columns required for estimate *ucol*
        :Versions:
            * 2020-03-23 ``@ddalle``: Version 1.0
            * 2020-05-08 ``@ddalle``: Remove if *acols* is ``None``
        """
        # Get dictionary of ecols
        uq_acols = self.__dict__.setdefault("uq_acols", {})
        # Check for trivial input
        if acols is None:
            uq_acols.pop(ucol, None)
            return
        # Check type
        if typeutils.isstr(acols):
            # Create single list
            acols = [acols]
        elif acols is None:
            # Empty result should be empty list
            acols = []
        elif not isinstance(acols, list):
            # Invalid type
            raise TypeError(
                "uq_acols for col '%s' should be list; got '%s'"
                % (ucol, type(acols)))
        # Set it
        uq_acols[ucol] = acols

    # Set aux functions for uq
    def set_uq_afunc(self, ucol, afunc):
        r"""Set function to UQ column if aux cols are present

        :Call:
            >>> db.set_uq_afunc(ucol, afunc)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *ucol*: :class:`str`
                Name of UQ col to estimate
            *afunc*: **callable**
                Function to estimate *ucol*
        :Versions:
            * 2020-03-23 ``@ddalle``: Version 1.0
            * 2020-05-08 ``@ddalle``: Remove if *afunc* is ``None``
        """
        # Get dictionary of aux UQ funcs
        uq_afuncs = self.__dict__.setdefault("uq_afuncs", {})
        # Check for ``None``
        if afunc is None:
            uq_afuncs.pop(ucol, None)
            return
        # Check input type
        if not callable(afunc):
            raise TypeError("Function is not callable")
        # Get entry for *col*
        uq_afuncs[ucol] = afunc
  # >

  # ===================
  # Increment & Deltas
  # ===================
  # <
   # --- Increment ---
    # Create increment and UQ estimate for one column
    def genr8_udiff_by_rbf(self, db2, cols, scol=None, **kw):
        r"""Generate increment and UQ estimate between two responses

        :Call:
            >>> ddb = db.genr8_udiff_by_rbf(db2, cols, scol=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *db2*: :class:`DataKit`
                Second data container
            *cols*: :class:`list`\ [:class:`str`]
                Data columns to analyze
            *scol*: {``None``} | :class:`str` | :class:`list`
                List of arguments to define slices on which to smooth
            *smooth*: {``0``} | :class:`float` >= 0
                Smoothing parameter for interpolation on slices
            *function*: {``"multiquadric"``} | :class:`str`
                RBF basis function type, see :func:`scirbf.Rbf`
            *test_values*: {``db``} | :class:`dict`
                Candidate values of each *arg* for differencing
            *test_bkpts*: {``None``} | :class:`dict`
                Candidate break points (1D unique values) to override
                *test_values*. Used to create full-factorial matrix.
            *tol*: {``1e-4``} | :class:`float` > 0
                Default tolerance for matching slice constraints
            *tols*: {``{}``} | :class:`dict` (:class:`float` >= 0)
                Specific tolerance for particular slice keys
        :Outputs:
            *ddb*: *db.__class__*
                New database with filtered *db* and *db2* diffs
            *ddb[arg]*: :class:`np.ndarray`
                Test values for each *arg* in *col* response args
            *ddb[col]*: :class:`np.ndarray`
                Smoothed difference between *db2* and *db*
            *ddb._slices*: :class:`list`\ [:class:`np.ndarray`]
                Saved lists of indices on which smoothing is performed
        :Versions:
            * 2020-05-08 ``@ddalle``: Version 1.0
        """
        # Create primary (smoothed) deltas
        ddb = self.genr8_rdiff_by_rbf(db2, cols, scol=scol, **kw)
        # Create raw deltas
        ddb0 = self.genr8_rdiff(db2, cols, **kw)
        # Copy any UQ cols
        for col in cols:
            # Get response args (guaranteed to be here)
            args = self.get_response_args(col)
            # Copy them, but set eval method to 'nearest' b/c
            # the differencing probably broke the orig response.
            # We need this to calculate UQ, but *ddb* and *ddb0*
            # have the same points
            ddb.make_response(col, "nearest", args)
            ddb0.make_response(col, "nearest", args)
            # Name of UQ column (if any)
            ucol = self.get_uq_col(col)
            # Check if any
            if ucol is None:
                continue
            # Copy it
            ddb.set_uq_col(col, ucol)
            # Get response for *ucol*
            uargs = self.get_response_args(ucol)
            umeth = self.get_response_method(ucol)
            # Copy them
            if uargs:
                ddb.set_response_args(ucol, uargs)
            if umeth:
                ddb.set_response_method(ucol, umeth)
            # Get aux cols to compute *ucol*
            acols = self.get_uq_acol(ucol)
            # Get extra cols related to *ucol*
            ecols = self.get_uq_ecol(ucol)
            # Check for *acols*
            if acols:
                # Set them
                ddb.set_uq_acol(ucol, acols)
                # Copy function
                ddb.set_uq_afunc(ucol, self.get_uq_afunc(ucol))
            # Check for *ecols*
            if ecols:
                # Set them
                ddb.set_uq_ecol(ucol, ecols)
                # Loop through extra cols
                for ecol in ecols:
                    # Copy function
                    ddb.set_uq_efunc(ecol, self.get_uq_efunc(ecol))
        # Estimate UQ
        ddb.est_uq_db(ddb0, cols, **kw)
        # Output
        return ddb

   # --- Pairwise Deltas ---
    # Generate deltas by smoothed response
    def genr8_rdiff_by_rbf(self, db2, cols, scol=None, **kw):
        r"""Generate smoothed deltas between two responses

        :Call:
            >>> ddb = db.genr8_rdiff_by_rbf(db2, cols, scol, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *db2*: :class:`DataKit`
                Second data container
            *cols*: :class:`list`\ [:class:`str`]
                Data columns to analyze
            *scol*: {``None``} | :class:`str` | :class:`list`
                List of arguments to define slices on which to smooth
            *smooth*: {``0``} | :class:`float` >= 0
                Smoothing parameter for interpolation on slices
            *function*: {``"multiquadric"``} | :class:`str`
                RBF basis function type, see :func:`scirbf.Rbf`
            *test_values*: {``db``} | :class:`dict`
                Candidate values of each *arg* for differencing
            *test_bkpts*: {``None``} | :class:`dict`
                Candidate break points (1D unique values) to override
                *test_values*. Used to create full-factorial matrix.
            *tol*: {``1e-4``} | :class:`float` > 0
                Default tolerance for matching slice constraints
            *tols*: {``{}``} | :class:`dict` (:class:`float` >= 0)
                Specific tolerance for particular slice keys
            *v*, *verbose*: ``True`` | {``False``}
                Verbose STDOUT flag
        :Outputs:
            *ddb*: *db.__class__*
                New database with filtered *db* and *db2* diffs
            *ddb[arg]*: :class:`np.ndarray`
                Test values for each *arg* in *col* response args
            *ddb[col]*: :class:`np.ndarray`
                Smoothed difference between *db2* and *db*
            *ddb._slices*: :class:`list`\ [:class:`np.ndarray`]
                Saved lists of indices on which smoothing is performed
        :Versions:
            * 2020-05-08 ``@ddalle``: Fork from :func:`DBCoeff.DiffDB`
        """
       # --- Options and Init ---
        # Create new instance
        ddb = self.__class__()
        # Verbosity option
        verbose = kw.get("verbose", kw.get("v", False))
        # Keyword args to pass to evaluation
        kw_response = kw.get("response_kwargs", {})
        # Get smoothing parameter
        smooth = kw.get("smooth", 0.0)
        # RBF basis function
        func = kw.get("function", "multiquadric")
        # Label format
        fmt = "Differencing %-40s\r" % ("%s slice %i/%i")
        # Reference column
        col = cols[0]
        # Evaluation arguments
        args = self.get_response_args(col)
        # Ensure that there are some
        if args is None:
            raise ValueError("No response args set for '%s'" % col)
        # Arg count
        narg = len(args)
        # Ensure list for *scol*
        if not scol:
            # Use empty list for empty slicing
            scols = []
        elif typeutils.isstr(scol):
            # Convert to list
            scols = [scol]
        elif not isinstance(scol, list):
            # Bad type
            raise TypeError("Slice col list 'scol' must be list")
        else:
            # Already list
            scols = scol
       # --- Test Points ---
        # Data type for output col
        dtype = self.get_col_dtype(col)
        # Get test values and slices
        vals, slices = self._genr8_test_slices(args, scol, **kw)
        # Reference arg
        arg0 = args[0]
        # Number of test points
        nx = vals[arg0].size
        # Init list of args that are interpolated (not slice args)
        rbf_args = []
        # Mask for RBF args
        mask_rbf_args = np.full(narg, True)
        # Loop through all args to check if slice args
        for j, arg in enumerate(args):
            # Check if it's a slice arg
            if arg in scol:
                # Not a slice arg
                mask_rbf_args[j] = False
            else:
                # Append it to interp list
                rbf_args.append(arg)
        # Number of interp args
        nargi = len(rbf_args)
       # --- Compute Diffs ---
        # Number of slices
        ns = len(slices)
        # Loop through columns
        for col in cols:
            # Initialize deltas
            dv = np.zeros(nx, dtype=dtype)
            # Loop through slices
            for j, J in enumerate(slices):
                # Status update
                if verbose:
                    sys.stdout.write(fmt % (col, j+1, ns))
                    sys.stdout.flush()
                # Number of points in slice
                nxj = J.size
                # Initialize matrix of evaluation points
                A = np.zeros((narg, nxj))
                # Initialize inputs to RBF
                X = []
                # Loop through args
                for k, arg in enumerate(args):
                    # Save data to row of *A*
                    A[k] = vals[arg][J]
                    # Save RBF inputs
                    if arg not in scol:
                        X.append(A[k])
                # Evaluate both databases
                v1 = self(col, *A, **kw_response)
                v2 = db2(col, *A, **kw_response)
                # Deltas
                dvj = v2 - v1
                # Check for valid smoothing
                if (nargi > 0) and (smooth > 0):
                    # Get inputs for RBF
                    R = X + [dvj]
                    # Create RBF
                    fn = scirbf.Rbf(*R, function=func, smooth=smooth)
                    # Save evaluated (smoothed) deltas
                    dv[J] = fn(*X)
                else:
                    # Raw deltas if no smoothing (or all args in *scols*)
                    dv[J] = dvj
            # Save definition for *col*
            ddb.set_defn(col, self.get_defn(col))
            # Save values
            ddb.save_col(col, dv)
       # --- Cleanup ---
        # Clean up prompt
        if verbose:
            sys.stdout.write("%72s\r" % "")
            sys.stdout.flush()
        # Save test values
        for arg in args:
            # Save definition
            ddb.set_defn(arg, self.get_defn(arg))
            # Save values
            ddb.save_col(arg, vals[arg])
       # --- Co-mapped XAargs ---
        # Trajectory co-keys
        cocols = kw.get("cocols", list(self.bkpts.keys()))
        # Check for a mapping column
        if len(scols) > 0:
            # Map on first slice *col*
            maincol = scols[0]
            # Get original values
            mainvals = self.get_all_values(maincol)
            # Unique values
            mainbkpts = self.get_bkpt(maincol)
        # Map other breakpoint keys
        for col in cocols:
            # Skip if already present
            if col in args:
                continue
            elif col in cols:
                continue
            # Check for slices
            if maincol is None:
                break
            # Get values for this column
            v0 = self.get_all_values(col)
            # Check if present
            if v0 is None:
                raise KeyError("No *cocol* called '%s' found" % col)
            # Check size
            if mainvals.size != v0.size:
                # Original sizes do not match; no map applicable
                continue
            # Output values of slice key
            x0 = ddb[maincol]
            # Initialize data
            v = np.zeros_like(x0)
            # Status update
            if verbose:
                sys.stdout.write("Mapping key '%s'\r" % col)
                sys.stdout.flush()
            # Loop through slice values
            for xi in mainbkpts:
                # Find value of slice key matching that parameter
                i = np.where(mainvals == xi)[0][0]
                # Output value
                vi = v0[i]
                # Get the indices of break points with that value
                J = np.where(x0 == xi)[0]
                # Evaluate coefficient
                v[J] = vi
            # Save the values
            ddb.save_col(col, v)
            # Copy definition
            ddb.set_defn(col, self.get_defn(col))
        # Clean up prompt
        if verbose:
            sys.stdout.write("%72s\r" % "")
            sys.stdout.flush()
       # --- Output ---
        # Save slices
        ddb._slices = slices
        # Output
        return ddb

    # Generate deltas by response
    def genr8_rdiff(self, db2, cols, **kw):
        r"""Generate deltas between responses of two databases

        :Call:
            >>> ddb = db.genr8_rdiff(db2, col, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *db2*: :class:`DataKit`
                Second data container
            *cols*: :class:`list`\ [:class:`str`]
                Data columns to difference
            *test_values*: {``db``} | :class:`dict`
                Candidate values of each *arg* for differencing
            *v*, *verbose*: ``True`` | {``False``}
                Verbose STDOUT flag
        :Outputs:
            *ddb*: *db.__class__*
                New database with filtered *db* and *db2* diffs
            *ddb[arg]*: :class:`np.ndarray`
                Test values for each *arg* in *col* response args
            *ddb[col]*: :class:`np.ndarray`
                Smoothed difference between *db2* and *db*
        :Versions:
            * 2020-05-08 ``@ddalle``: Version 1.0
        """
        # Create new instance
        ddb = self.__class__()
        # Keyword args to pass to evaluation
        kw_response = kw.get("response_kwargs", {})
        # Verbosity option
        verbose = kw.get("verbose", kw.get("v", False))
        # Label format
        fmt = "Differencing %-40s\r" % ("%s")
        # Reference column
        col = cols[0]
        # Evaluation arguments
        args = self.get_response_args(col)
        # Ensure that there are some
        if args is None:
            raise ValueError("No response args set for '%s'" % col)
        # Arg count
        narg = len(args)
        # Data type for output col
        dtype = self.get_col_dtype(col)
        # Get test values and slices
        vals, _ = self._get_test_values(args, **kw)
        # Reference arg
        arg0 = args[0]
        # Number of test points
        nx = vals[arg0].size
        # Initialize matrix of evaluation points
        A = np.zeros((narg, nx))
        # Loop through args
        for k, arg in enumerate(args):
            # Save data to row of *A*
            A[k] = vals[arg]
            # Save arg
            ddb.save_col(arg, vals[arg])
            # Copy definition
            ddb.set_defn(arg, self.get_defn(arg))
        # Loop through columns
        for col in cols:
            # Status update
            if verbose:
                sys.stdout.write(fmt % col)
                sys.stdout.flush()
            # Evaluate both databases
            v1 = self(col, *A, **kw_response)
            v2 = db2(col, *A, **kw_response)
            # Save difference
            ddb.save_col(col, v2 - v1)
            # Link definition
            ddb.set_defn(col, self.get_defn(col))
        # Clean up prompt
        if verbose:
            sys.stdout.write("%72s\r" % "")
            sys.stdout.flush()
        # Output
        return ddb

   # --- Conditions and Slices ---
    # Generate slices for interpolation
    def _genr8_test_slices(self, args, scols, **kw):
        r"""Get test values for creating windows or comparing databases

        :Call:
            >>> vals, slices = db._genr8_test_slices(args, scols, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *args*: :class:`list`\ [:class:`str`]
                List of arguments to use for windowing
            *scols*: :class:`list`\ [:class:`str`]
                List of "slice" args held constant for each slice
        :Keyword Arguments:
            *test_values*: {*db*} | :class:`DataKit` | :class:`dict`
                Specify values of each parameter in *args* that are the
                candidate points for the window; default is from *db*
            *test_bkpts*: {``None``} | :class:`dict`
                Specify candidate window boundaries; must be ascending
                array of unique values for each key.  If specified,
                overrides *test_values*
            *tol*: {``1e-4``} | :class:`float` >= 0
                Default tolerance for all slice keys
            *tols*: {``{}``} | :class:`dict` (:class:`float` >= 0)
                Specific tolerance for particular slice keys
        :Outputs:
            *vals*: :class:`dict`\ [:class:`np.ndarray`]
                Dictionary of lookup values for each *arg* in *args*
            *slices*: :class:`list`\ [:class:`np.ndarray`]
                Indices (relative to *vals*) of points in each slice
        :Versions:
            * 2019-02-20 ``@ddalle``: Version 1.0
            * 2020-05-07 ``@ddalle``: Fork from :class:`db1.DBCoeff`
        """
       # --- Tolerances ---
        # Initialize tolerances
        tols = dict(kw.get("tols", {}))
        # Default global tolerance
        tol = kw.get("tol", 1e-4)
       # --- Test values ---
        # Get values
        vals, bkpts = self._get_test_values(args, **kw)
        # Number of values
        nv = vals[args[0]].size
       # --- Slice candidates ---
        # Create tuple of all breakpoint combinations for slice keys
        bkpts1d = tuple(bkpts[col] for col in scols)
        # Create *n*-dimensional array of points (full factorial)
        vals_nd = np.meshgrid(*bkpts1d, indexing="ij")
        # Create matrix of potential slice coordinates
        slice_vals = {}
        # Default number of candidate slices: 1 if no slice keys
        ns = 1
        # Loop through slice keys
        for (j, col) in enumerate(scols):
            # Save the flattened array
            slice_vals[col] = vals_nd[j].flatten()
            # Number of possible of values
            ns = slice_vals[col].size
       # --- Slice indices ---
        # Initialize slices [array[int]]
        slices = []
        # Loop through potential slices
        for j in range(ns):
            # Initialize indices meeting all constraints
            M = np.arange(nv) > -1
            # Loop through slice arguments
            for col in scols:
                # Get key value
                v = slice_vals[col][j]
                # Get tolerance
                tolk = tols.get(col, tol)
                # Apply constraint
                M = np.logical_and(M, np.abs(vals[col] - v) <= tolk)
            # Convert *M* to indices
            I = np.where(M)[0]
            # Check for matches
            if I.size == 0:
                continue
            # Save slice
            slices.append(I)
        # Output
        return vals, slices
  # >

  # ===================
  # Break Points
  # ===================
  # <
   # --- Breakpoint Creation ---
    # Get automatic break points
    def create_bkpts(self, cols, nmin=5, tol=1e-12, tols={}, mask=None):
        r"""Create automatic list of break points for interpolation

        :Call:
            >>> db.create_bkpts(col, nmin=5, tol=1e-12, **kw)
            >>> db.create_bkpts(cols, nmin=5, tol=1e-12, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *col*: :class:`str`
                Individual lookup variable
            *cols*: :class:`list`\ [:class:`str`]
                List of lookup variables
            *nmin*: {``5``} | :class:`int` > 0
                Minimum number of data points at one value of a key
            *tol*: {``1e-12``} | :class:`float` >= 0
                Tolerance for values considered to be equal
            *tols*: {``{}``} | :class:`dict`\ [:class:`float`]
                Tolerances for specific *cols*
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Mask of which database indices to consider
        :Outputs:
            *db.bkpts*: :class:`dict`
                Dictionary of 1D unique lookup values
            *db.bkpts[col]*: :class:`np.ndarray` | :class:`list`
                Unique values of *DBc[col]* with at least *nmin* entries
        :Versions:
            * 2018-06-08 ``@ddalle``: Version 1.0
            * 2019-12-16 ``@ddalle``: Updated for :mod:`rdbnull`
            * 2020-03-26 ``@ddalle``: Renamed, :func:`get_bkpts`
            * 2020-05-06 ``@ddalle``: Moved much to :func:`genr8_bkpts`
        """
        # Check for single key list
        if not isinstance(cols, (list, tuple)):
            # Make list
            cols = [cols]
        # Filter specific tolerances
        if tols is None:
            tols = {}
        # Initialize break points
        bkpts = self.__dict__.setdefault("bkpts", {})
        # Loop through keys
        for col in cols:
            # Get tolerance
            ctol = tols.get(col, tol)
            # Save these break points
            bkpts[col] = self.genr8_bkpts(col, nmin=nmin, tol=ctol, mask=mask)

    # Get break points for specific col
    def genr8_bkpts(self, col, nmin=5, tol=1e-12, mask=None):
        r"""Generate list of unique values for one *col*

        :Call:
            >>> B = db.genr8_bkpts(col, nmin=5, tol=1e-12, mask=None)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *col*: :class:`str`
                Individual lookup variable
            *nmin*: {``5``} | :class:`int` > 0
                Minimum number of data points at one value of a key
            *tol*: {``1e-12``} | :class:`float` >= 0
                Tolerance for values considered to be equal
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Mask of which database indices to consider
        :Outputs:
            *B*: :class:`np.ndarray` | :class:`list`
                Unique values of *DBc[col]* with at least *nmin* entries
        :Versions:
            * 2020-05-06 ``@ddalle``: Version 1.0
        """
        # Check type
        if not isinstance(col, typeutils.strlike):
            raise TypeError("Column name is not a string")
        # Check if present
        if col not in self.cols:
            raise KeyError("Lookup column '%s' is not present" % col)
        # Get all values
        V = self.get_values(col, mask)
        # Get data type
        dtype = self.get_col_dtype(col)
        # Check dtype
        if dtype == "str":
            # Get unique values without converting to array
            B = list(set(V))
        elif dtype.startswith("int"):
            # No need to apply tolerance
            B = np.unique(V)
        else:
            # Get unique values of array
            U = np.unique(V)
            # Initialize filtered value
            B = np.zeros_like(U)
            n = 0
            # Loop through entries
            for v in U:
                # Check if too close to a previous entry
                if (n > 0) and np.min(np.abs(v - B[:n])) <= tol:
                    # Close to previous "unique" value
                    continue
                # Count entries
                if np.count_nonzero(np.abs(V - v) <= tol) >= nmin:
                    # Save the value
                    B[n] = v
                    # Increase count
                    n += 1
            # Trim
            B = B[:n]
        # Output
        return B

    # Map break points from other key
    def create_bkpts_map(self, cols, scol, tol=1e-12):
        r"""Map break points of one column to one or more others

        The most common purpose to use this method is to create
        non-ascending break points.  One common example is to keep track
        of the dynamic pressure values at each Mach number.  These
        dynamic pressures may be unique, but sorting them by dynamic
        pressure is different from the order in which they occur in
        flight.

        :Call:
            >>> db.create_bkpts_map(cols, scol, tol=1e-12)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *cols*: :class:`list`\ [:class:`str`]
                Individual lookup variable
            *scol*: :class:`str`
                Name of key to drive map/schedule
            *tol*: {``1e-12``} | :class:`float` >= 0
                Tolerance cutoff (used for *scol*)
        :Outputs:
            *DBc.bkpts*: :class:`dict`
                Dictionary of 1D unique lookup values
            *DBc.bkpts[key]*: :class:`np.ndarray`\ [:class:`float`]
                Unique values of *DBc[key]* with at least *nmin* entries
        :Versions:
            * 2018-06-29 ``@ddalle``: Version 1.0
            * 2019-12-16 ``@ddalle``: Ported to :mod:`rdbnull`
            * 2020-03-26 ``@ddalle``: Renamed, :func:`map_bkpts`
        """
        # Check inputs
        if not isinstance(cols, list):
            raise TypeError("Columns input must be a list")
        elif not isinstance(scol, typeutils.strlike):
            raise TypeError("Schedule key must be a string")
        # Get break points
        bkpts = self.__dict__.get("bkpts")
        # Check break points for *scol*
        if bkpts is None:
            raise AttributeError("No 'bkpts' attribute; call get_bkpts()")
        elif scol not in bkpts:
            raise AttributeError("No bkpts for col '%s'" % col)
        # Get data type of *scol*
        dtype = self.get_col_dtype(scol)
        # Check data type
        if not (dtype.startswith("float") or dtype.startswith("int")):
            raise TypeError(
                ("Schedule col '%s' must have either " % scol) +
                ("float or int type (got %s)" % dtype))
        # Get schedule break points and nominal values
        V0 = self[scol]
        U0 = self.bkpts[scol]
        # Loop through keys
        for col in cols:
            # Check type
            if not isinstance(col, typeutils.strlike):
                raise TypeError("Column name is not a string")
            # Check if present
            if col not in self.cols:
                raise KeyError("Lookup column '%s' is not present" % col)
            # Get data type
            dtype = self.get_col_dtype(col)
            # Check it
            if not (dtype.startswith("float") or dtype.startswith("int")):
                raise TypeError(
                    ("Break point col '%s' must have either " % col) +
                    ("float or int type (got %s)" % dtype))
            # Values for *col*
            V = self[col]
            # Initialize break point array for *col*
            U = np.zeros(U0.size, dtype=dtype)
            # Check shape
            if V.size != V0.size:
                raise ValueError(
                    ("Col '%s' (%i) has different size " % (col, V.size)) +
                    ("from schedule col '%s' (%i)" % (scol, V0.size)))
            # Loop through slice values
            for (j, v0) in enumerate(U0):
                # Find value of slice key matching that parameter
                i = np.where(np.abs(V0 - v0) <= tol)[0][0]
                # Save value of that index from col
                U[j] = V[i]
            # Save break points
            bkpts[col] = U

    # Schedule break points at slices at other key
    def create_bkpts_schedule(self, cols, scol, nmin=5, tol=1e-12):
        r"""Create lists of unique values at each unique value of *scol*

        This function creates a break point list of the unique values of
        each *col* in *cols* at each unique value of a "scheduling"
        column *scol*.  For example, if a different run matrix of
        *alpha* and *beta* is used at each *mach* number, this function
        creates a list of the unique *alpha* and *beta* values for each
        Mach number in *db.bkpts["mach"]*.

        :Call:
            >>> db.create_bkpts_schedule(cols, scol)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *cols*: :class:`list`\ [:class:`str`]
                Individual lookup variable
            *scol*: :class:`str`
                Name of key to drive map/schedule
            *nmin*: {``5``} | :class:`int` > 0
                Minimum number of data points at one value of a key
            *tol*: {``1e-12``} | :class:`float` >= 0
                Tolerance cutoff
        :Outputs:
            *db.bkpts*: :class:`dict`
                Dictionary of unique lookup values
            *db.bkpts[col]*: :class:`list`\ [:class:`np.ndarray`]
                Unique values of *db[col]* at each value of *scol*
        :Versions:
            * 2018-06-29 ``@ddalle``: Version 1.0
            * 2019-12-16 ``@ddalle``: Ported to :mod:`rdbnull`
            * 2020-03-26 ``@ddalle``: Renamed, :func:`schedule_bkpts`
        """
        # Check inputs
        if not isinstance(cols, list):
            raise TypeError("Columns input must be a list")
        elif not isinstance(scol, typeutils.strlike):
            raise TypeError("Schedule key must be a string")
        # Get break points
        bkpts = self.__dict__.get("bkpts")
        # Check break points for *scol*
        if bkpts is None:
            raise AttributeError("No 'bkpts' attribute; call get_bkpts()")
        elif scol not in bkpts:
            raise AttributeError("No bkpts for col '%s'" % col)
        # Get data type of *scol*
        dtype = self.get_col_dtype(scol)
        # Check data type
        if not (dtype.startswith("float") or dtype.startswith("int")):
            raise TypeError(
                ("Schedule col '%s' must have either " % scol) +
                ("float or int type (got %s)" % dtype))
        # Get schedule break points and nominal values
        V0 = self[scol]
        U0 = self.bkpts[scol]
        # Loop through keys
        for col in cols:
            # Check type
            if not isinstance(col, typeutils.strlike):
                raise TypeError("Column name is not a string")
            # Check if present
            if col not in self.cols:
                raise KeyError("Lookup column '%s' is not present" % col)
            # Get data type
            dtype = self.get_col_dtype(col)
            # Check it
            if not (dtype.startswith("float") or dtype.startswith("int")):
                raise TypeError(
                    ("Break point col '%s' must have either " % col) +
                    ("float or int type (got %s)" % dtype))
            # Values for *col*
            V = self[col]
            # Initialize break point array for *col*
            U = np.zeros(U0.size, dtype=dtype)
            # Check shape
            if V.size != V0.size:
                raise ValueError(
                    ("Col '%s' (%i) has different size " % (col, V.size)) +
                    ("from schedule col '%s' (%i)" % (scol, V0.size)))
            # Initialize scheduled break points
            X = []
            # Get all values for this key
            V = self[col]
            # Loop through slice values
            for (j, v0) in enumerate(U0):
                # Indices of points in this slice
                I = np.where(np.abs(V0 - v0) <= tol)[0]
                # Check for broken break point
                if I.size == 0:
                    # This shouldn't happen
                    raise ValueError("No points matching slice at " +
                        ("%s = %.2e" % (scol, v0)))
                elif I.size < nmin:
                    # No hope of break points at this *scol* value
                    X.append(np.zeros(0, dtype=V.dtype))
                    continue
                # Get unique values on the slice
                U = np.unique(V[I])
                # Initialize filtered value
                B = np.zeros_like(U)
                n = 0
                # Loop through entries
                for v in U:
                    # Check if too close to a previous entry
                    if (n > 0) and np.min(np.abs(v - B[:n])) <= tol:
                        continue
                    # Count entries
                    if np.count_nonzero(np.abs(V[I] - v) <= tol) >= nmin:
                        # Save the value
                        B[n] = v
                        # Increment count
                        n += 1
                # Save break point
                X.append(B[:n])
            # Save break points
            bkpts[col] = X

   # --- Breakpoint Lookup ---
    # Find index of break point value
    def get_bkpt_index(self, col, v, tol=1e-8):
        r"""Get interpolation weights for 1D linear interpolation

        :Call:
            >>> i0, i1, f = db.get_bkpt_index(k, v, tol=1e-8)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *col*: :class:`str`
                Individual lookup variable from *db.bkpts*
            *v*: :class:`float`
                Value at which to lookup
            *tol*: {``1e-8``} | :class:`float` >= 0
                Tolerance for left and right bounds
        :Outputs:
            *i0*: ``None`` | :class:`int`
                Lower bound index, if ``None``, extrapolation below
            *i1*: ``None`` | :class:`int`
                Upper bound index, if ``None``, extrapolation above
            *f*: 0 <= :class:`float` <= 1
                Lookup fraction, ``1.0`` if *v* is equal to upper bound
        :Versions:
            * 2018-12-30 ``@ddalle``: Version 1.0
            * 2019-12-16 ``@ddalle``: Updated for :mod:`rdbnull`
        """
        # Extract values
        try:
            # Naive extractions
            V =self.bkpts[col]
        except AttributeError:
            # No break points
            raise AttributeError("No break point dict present")
        except KeyError:
            # Missing key
            raise KeyError(
                "Col '%s' is not present in break point dict" % col)
        # Check type
        if not isinstance(V, np.ndarray):
            # Bad break point array type
            raise TypeError(
                "Break point list for '%s' is not np.ndarray" % col)
        elif V.ndim != 1:
            # Multidmensional array
            raise ValueError(
                ("Cannot perform lookup on %iD array " % V.ndim) +
                ("for column '%s'" % col))
        elif V.size == 0:
            # No break points
            raise ValueError("Break point array for col '%s' is empty" % col)
        elif V.size == 1:
            # Only one
            raise ValueError(
                "Break point array for col '%s' has only one entry" % col)
        # Output
        return self._bkpt_index(V, v, tol=tol)

    # Function to get interpolation weights for uq
    def get_bkpt_index_schedule(self, k, v, j):
        """Get weights 1D interpolation of *k* at a slice of master key

        :Call:
            >>> i0, i1, f = db.get_bkpt_index_schedule(k, v, j)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *k*: :class:`str`
                Name of trajectory key in *FM.bkpts* for lookup
            *v*: :class:`float`
                Value at which to lookup
            *j*: :class:`int`
                Index of master "slice" key, if *k* has scheduled
                break points
        :Outputs:
            *i0*: ``None`` | :class:`int`
                Lower bound index, if ``None``, extrapolation below
            *i1*: ``None`` | :class:`int`
                Upper bound index, if ``None``, extrapolation above
            *f*: 0 <= :class:`float` <= 1
                Lookup fraction, ``1.0`` if *v* is equal to upper bound
        :Versions:
            * 2018-04-19 ``@ddalle``: Version 1.0
        """
        # Get potential values
        V = self._scheduled_bkpts(k, j)
        # Lookup within this vector
        return self._bkpt(V, v)

    # Get break point from vector
    def _bkpt_index(self, V, v, tol=1e-5, col=None):
        r"""Get interpolation weights for 1D interpolation

        This function tries to find *i0* and *i1* such that *v* is
        between *V[i0]* and *V[i1]*.  It assumes the values of *V* are
        unique and ascending (not checked).

        :Call:
            >>> i0, i1, f = db._bkpt_index(V, v, tol=1e-8)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *V*: :class:`np.ndarray`\ [:class:`float`]
                1D array of data values
            *v*: :class:`float`
                Value at which to lookup
            *tol*: {``1e-8``} | :class:`float` >= 0
                Tolerance for left and right bounds
        :Outputs:
            *i0*: ``None`` | :class:`int`
                Lower bound index, if ``None``, extrapolation below
            *i1*: ``None`` | :class:`int`
                Upper bound index, if ``None``, extrapolation above
            *f*: 0 <= :class:`float` <= 1
                Lookup fraction, ``1.0`` if *v* is equal to upper bound;
                can be outside 0-1 bound for extrapolation
        :Versions:
            * 2018-12-30 ``@ddalle``: Version 1.0
            * 2019-12-16 ``@ddalle``: Version 2.0; for :mod:`rdbnull`
        """
        # Get length
        n = V.size
        # Get min/max
        vmin = np.min(V)
        vmax = np.max(V)
        # Check for extrapolation cases
        if n == 1:
            # Only one point
            return 0, None, 1.0
        if v < vmin - tol*(vmax-vmin):
            # Extrapolation left
            return None, 0, (v-V[0])/(V[1]-V[0])
        if v > vmax + tol*(vmax-vmin):
            # Extrapolation right
            return n-1, None, (v-V[-2])/(V[-1]-V[-2])
        # Otherwise, count up values below
        i0 = np.sum(V[:-1] <= v) - 1
        i1 = i0 + 1
        # Progress fraction
        f = (v - V[i0]) / (V[i1] - V[i0])
        # Output
        return i0, i1, f

    # Get a break point, with error checking
    def get_bkpt(self, col, *I):
        r"""Extract a breakpoint by index, with error checking

        :Call:
            >>> v = db.get_bkpt(col, *I)
            >>> v = db.get_bkpt(col)
            >>> v = db.get_bkpt(col, i)
            >>> v = db.get_bkpt(col, i, j)
            >>> v = db.get_bkpt(col, i, j, ...)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *col*: :class:`str`
                Individual lookup variable from *db.bkpts*
            *I*: :class:`tuple`
                Tuple of lookup indices
            *i*: :class:`int`
                (Optional) first break point list index
            *j*: :class:`int`
                (Optional) second break point list index
        :Outputs:
            *v*: :class:`float` | :class:`np.ndarray`
                Break point or array of break points
        :Versions:
            * 2018-12-31 ``@ddalle``: Version 1.0
            * 2019-12-16 ``@ddalle``: Updated for :mod:`rdbnull`
        """
        # Get the break points
        try:
            v = self.bkpts[col]
        except AttributeError:
            # No radial basis functions at all
            raise AttributeError("No break points found")
        except KeyError:
            # No RBF for this coefficient
            raise KeyError("No break points for col '%s'" % col)
        # Number of indices given
        nd = len(I)
        # Loop through indices
        for n, i in enumerate(I):
            # Try to extract
            try:
                # Get the *ith* list entry
                v = v[i]
            except (IndexError, TypeError):
                # Reached scalar too soon
                raise TypeError(
                    ("Breakpoints for '%s':\n" % col) +
                    ("Expecting %i-dimensional " % nd) +
                    ("array but found %i-dim" % n))
        # Output
        return v

    # Get all break points
    def _scheduled_bkpts(self, col, j):
        """Get list of break points for key *col* at schedule *j*

        :Call:
            >>> i0, i1, f = db._scheduled_bkpts(col, j)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *col*: :class:`str`
                Individual lookup variable from *db.bkpts*
            *j*: :class:`int`
                Index of master "slice" key, if *col* has scheduled
                break points
        :Outputs:
            *i0*: ``None`` | :class:`int`
                Lower bound index, if ``None``, extrapolation below
            *i1*: ``None`` | :class:`int`
                Upper bound index, if ``None``, extrapolation above
            *f*: 0 <= :class:`float` <= 1
                Lookup fraction, ``1.0`` if *v* is equal to upper bound
        :Versions:
            * 2018-12-30 ``@ddalle``: Version 1.0
            * 2019-12-16 ``@ddalle``: Updated for :mod:`rdbnull`
        """
        # Get the break points
        try:
            V = self.bkpts[col]
        except AttributeError:
            # No radial basis functions at all
            raise AttributeError("No break points found")
        except KeyError:
            # No RBF for this coefficient
            raise KeyError("No break points for col '%s'" % col)
        # Get length
        n = len(V)
        # Size check
        if n == 0:
            raise ValueError("Empty break point array for col '%s'" % col)
        # Check first key for array
        if isinstance(V[0], (np.ndarray, list)):
            # Get break points for this slice
            V = V[j]
            # Reset size
            n = V.size
            # Recheck size
            if n == 0:
                raise ValueError(
                    ("Empty break point array for col '%s' " % col) +
                    ("at slice %i" % j))
        # Output
        return V

   # --- Full Factorial ---
    # Fill out a slice matrix
    def get_fullfactorial(self, scol=None, cols=None):
        r"""Create full-factorial matrix of values in break points

        This allows some of the break points cols to be scheduled, i.e.
        there are different matrices of *cols* for each separate value
        of *scol*.

        :Call:
            >>> X, slices = db.get_fullfactorial(scol=None, cols=None)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *scol*: {``None``} | :class:`str` | :class:`list`
                Optional name of slicing col(s)
            *cols*: {``None``} | :class:`list`\ [:class:`str`]
                List of (ordered) input keys, default is from *DBc.bkpts*
        :Outputs:
            *X*: :class:`dict`
                Dictionary of full-factorial matrix
            *slices*: :class:`dict` (:class:`ndarray`)
                Array of slice values for each col in *scol*
        :Versions:
            * 2018-11-16 ``@ddalle``: Version 1.0
        """
       # --- Slice col Checks ---
        # Check for list or string
        if isinstance(scol, list):
            # Get additional slice keys
            subcols = scol[1:]
            # Single slice key
            maincol = scol[0]
        elif scol is None:
            # No slices at all
            subcols = []
            maincol = None
        elif typeutils.isstr(scol):
            # No additional slice keys
            subcols = []
            maincol = scol
            # List of slice keys
            scol = [scol]
        else:
            raise TypeError("Slicing column must be 'list' or 'str'")
       # --- col Checks ---
        # Get break points
        bkpts = self.__dict__.get("bkpts", {})
        # Default col list
        if cols is None:
            # Default list
            cols = []
            # Loop through breakpoints
            for (col, V) in bkpts.items():
                # Check if *V* is an array
                if not isinstance(V, np.ndarray):
                    # Non-array
                    continue
                elif V.size == 0:
                    # Empty break points
                    continue
                elif V.ndim != 1:
                    # What? ND array
                    continue
                elif not isinstance(V[0], float):
                    # Not a simple number
                    continue
                # If reaching this point, usable column
                cols.append(col)
        else:
            # Loop through breakpoints
            for col in cols:
                # Get break points
                V = bkpts.get(col)
                # Check if *V* is an array
                if V is None:
                    raise KeyError("No breakpoints for col '%s'" % col)
                elif col in subcols and isinstance(V, (list, np.ndarray)):
                    # Special case for secondary slice col
                    continue
                elif not isinstance(V, np.ndarray):
                    # Non-array
                    raise TypeError(
                        "Breakpoints for col '%s' is not array" % col)
                elif V.size == 0:
                    # Empty break points
                    raise IndexError(
                        "Breakpoints for col '%s' is empty" % col)
                elif V.ndim != 1:
                    # What? ND array
                    raise IndexError(
                        "Breakpoints for col '%s' is not 1D" % col)
                elif not isinstance(V[0], (float, int, complex)):
                    # Not a simple number
                    raise TypeError(
                        "Non-numeric breakpoints for col '%s'" % col)
            # Make a copy
            cols = list(cols)
        # Eliminate *skey* if in key list
        if maincol in cols:
            cols.remove(maincol)
       # --- Slice Init ---
        # Initialize slice dictionary
        slices = {}
        # Initialize main slice values
        if maincol is not None:
            slices[maincol] = np.zeros(0)
        # Loop through slice keys
        for col in subcols:
            # Initialize slice
            slices[col] = np.zeros(0)
            # Check if listed in "cols"
            if col not in cols:
                cols.insert(0, col)
        # Number of columns
        ncol = len(cols)
        # Number of slice keys
        if scol is None:
            # No slices
            nscol = 0
        else:
            # Get length
            nscol = len(scol)
       # --- Matrix Init ---
        # Initialize dictionary of full-factorial matrix
        X = {}
        # Slice check
        if maincol is None:
            # No values to check
            M = np.zeros(1)
        else:
            # Get breakpoints for specified value
            M = bkpts[maincol]
            # Also keep track of slice key values
            X[maincol] = np.zeros(0)
        # Initialize values
        for col in cols:
            X[col] = np.zeros(0)
       # --- Main Slice Loop ---
        # Loop through slice values
        for (im, m) in enumerate(M):
            # Initialize matrix for this slice
            Xm = {}
            # Initialize slice values for this slice
            Xs = {}
            # Set slice values
            if maincol:
                # Main slice col has value of main col
                Xs[maincol] = np.array([m])
            # Copy values
            for col in cols:
                # Get values
                Vm = bkpts[col]
                # Get first entry for type checks
                v0 = bkpts[col][0]
                # Check if it's a scheduled key; will be an array
                if isinstance(v0, (list, np.ndarray)):
                    # Get break points for this slice key value
                    Vm = Vm[im]
                # Save the values
                Xm[col] = Vm
                # Save slice if appropriate
                if col in subcols:
                    Xs[col] = Vm
            # Loop through break point keys to create full-factorial inputs
            for i in range(1, ncol):
                # Name of first key
                col1 = cols[i]
                # Loop through keys 0 to *i*-1
                for j in range(i):
                    # Name of second key
                    col2 = cols[j]
                    # Create N+1 dimensional interpolation
                    x1, x2 = np.meshgrid(Xm[col1], Xm[col2])
                    # Flatten
                    Xm[col2] = x2.flatten()
                    # Save first key if *j* ix 0
                    if j == i-1:
                        Xm[col1] = x1.flatten()
            # Loop through slice keys to create full-factorial inputs
            for i in range(1, nscol):
                # Name of first key
                col1 = scol[i]
                # Loop through keys 0 to *i*-1
                for j in range(i):
                    # Name of second key
                    col2 = scol[j]
                    # Create N+1 dimensional interpolation
                    x1, x2 = np.meshgrid(Xs[col1], Xs[col2])
                    # Flatten
                    Xs[col2] = x2.flatten()
                    # Save first key if *j* ix 0
                    if j == i-1:
                        Xs[col1] = x1.flatten()
            # Save values
            for col in cols:
                X[col] = np.hstack((X[col], Xm[col]))
            # Process slices
            if maincol is not None:
                # Append to *scol* matrix
                X[maincol] = np.hstack(
                    (X[maincol], m*np.ones_like(Xm[col])))
                # Save slice full-factorial matrix
                for col in scol:
                    slices[col] = np.hstack((slices[col], Xs[col]))
        # Output
        return X, slices
  # >

  # ====================
  # Interpolation Tools
  # ====================
  # <
   # --- RBF construction ---
    # Regularization
    def create_global_rbfs(self, cols, args, I=None, **kw):
        r"""Create global radial basis functions for one or more columns

        :Call:
            >>> db.create_global_rbfs(cols, args, I=None)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *cols*: :class:`list`\ [:class:`str`]
                List of columns to create RBFs for
            *args*: :class:`list`\ [:class:`str`]
                List of (ordered) input keys, default is from *db.bkpts*
            *I*: {``None``} | :class:`np.ndarray`
                Indices of cases to include in RBF (default is all)
            *function*: {``"cubic"``} | :class:`str`
                Radial basis function type
            *smooth*: {``0.0``} | :class:`float` >= 0
                Smoothing factor, ``0.0`` for exact interpolation
        :Effects:
            *db.rbf[col]*: :class:`scipy.interpolate.rbf.Rbf`
                Radial basis function for each *col* in *cols*
        :Versions:
            * 2019-01-01 ``@ddalle``: Version 1.0
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
            * 2020-02-22 ``@ddalle``: Utilize :func:`create_rbf`
        """
        # Check for module
        if scirbf is None:
            raise ImportError("No scipy.interpolate.rbf module")
        # Create *rbf* attribute if needed
        rbf = self.__dict__.setdefault("rbf", {})
        # Loop through coefficients
        for col in cols:
            # Eval arguments for status update
            txt = str(tuple(args)).replace(" ", "")
            # Trim if too long
            if len(txt) > 50:
                txt = txt[:45] + "...)"
            # Status update line
            txt = "Creating RBF for %s%s" % (col, txt)
            sys.stdout.write("%-72s\r" % txt)
            sys.stdout.flush()
            # Create a single RBF
            rbf[col] = self.genr8_rbf(col, args, I=I, **kw)
        # Clean up the prompt
        sys.stdout.write("%72s\r" % "")
        sys.stdout.flush()

    # RBFs on slices
    def create_slice_rbfs(self, cols, args, I=None, **kw):
        r"""Create radial basis functions for each slice of *args[0]*

        The first entry in *args* is interpreted as a "slice" key; RBFs
        will be constructed at constant values of *args[0]*.

        :Call:
            >>> db.create_slice_rbfs(coeffs, args, I=None)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *cols*: :class:`list`\ [:class:`str`]
                List of columns to create RBFs for
            *args*: :class:`list`\ [:class:`str`]
                List of (ordered) input keys, default is from *db.bkpts*
            *I*: {``None``} | :class:`np.ndarray`
                Indices of cases to include in RBF (default is all)
            *function*: {``"cubic"``} | :class:`str`
                Radial basis function type
            *smooth*: {``0.0``} | :class:`float` >= 0
                Smoothing factor, ``0.0`` for exact interpolation
        :Effects:
            *db.rbf[col]*: :class:`list`\ [:class:`scirbf.Rbf`]
                List of RBFs at each slice for each *col* in *cols*
        :Versions:
            * 2019-01-01 ``@ddalle``: Version 1.0
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Check for module
        if scirbf is None:
            raise ImportError("No scipy.interpolate.rbf module")
        # Create *rbf* attribute if needed
        self.__dict__.setdefault("rbf", {})
        # RBF options
        func  = kw.get("function", "cubic")
        smooth = kw.get("smooth", 0.0)
        # Name of slice key
        skey = args[0]
        # Tolerances
        tols = kw.get("tols", {})
        # Tolerance for slice key
        tol = kw.get("tol", tols.get(skey, 1e-6))
        # Default indices
        if I is None:
            # Size of database
            n = len(self[skey])
            # All the indices
            I = np.arange(n)
        # Get break points for slice key
        B = self.bkpts[skey]
        # Number of slices
        nslice = len(B)
        # Initialize the RBFs
        for col in cols:
            self.rbf[col] = []
        # Loop through slices
        for b in B:
            # Get slice constraints
            qj = np.abs(self[skey][I] - b) <= tol
            # Select slice and add to list
            J = I[qj]
            # Create tuple of input points
            V = tuple(self[k][J] for k in args[1:])
            # Create a string for slice coordinate and remaining args
            arg_string_list = ["%s=%g" % (skey,b)]
            arg_string_list += [str(k) for k in args[1:]]
            # Joint list with commas
            arg_string = "(" + (",".join(arg_string_list)) + ")"
            # Loop through coefficients
            for col in cols:
                # Status update
                txt = "Creating RBF for %s%s" % (col, arg_string)
                sys.stdout.write("%-72s\r" % txt[:72])
                sys.stdout.flush()
                # Append reference values to input tuple
                Z = V + (self[col][J],)
                # Create a single RBF
                f = scirbf.Rbf(*Z, function=func, smooth=smooth)
                # Save it
                self.rbf[col].append(f)
        # Save break points for slice key
        self.bkpts[skey] = B
        # Save break points for other args
        self.create_bkpts_schedule(args[1:], skey, nmin=1)
        # Clean up the prompt
        sys.stdout.write("%72s\r" % "")
        sys.stdout.flush()

    # Individual RBF generator
    def genr8_rbf(self, col, args, I=None, **kw):
        r"""Create global radial basis functions for one or more columns

        :Call:
            >>> rbf = db.genr8_rbf(col, args, I=None)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`list`\ [:class:`str`]
                Data column to create RBF for
            *args*: :class:`list`\ [:class:`str`]
                List of (ordered) input cols
            *I*: {``None``} | :class:`np.ndarray`
                Indices of cases to include in RBF (default is all)
            *function*: {``"cubic"``} | :class:`str`
                Radial basis function type
            *smooth*: {``0.0``} | :class:`float` >= 0
                Smoothing factor, ``0.0`` for exact interpolation
        :Output:
            *rbf*: :class:`scipy.interpolate.rbf.Rbf`
                Radial basis function for *col*
        :Versions:
            * 2019-01-01 ``@ddalle``: Version 1.0
            * 2019-12-17 ``@ddalle``: Ported from :mod:`tnakit`
            * 2020-02-22 ``@ddalle``: Single-*col* version
            * 2020-03-06 ``@ddalle``: Name from :funC:`create_rbf`
        """
        # Check for module
        if scirbf is None:
            raise ImportError("No scipy.interpolate.rbf module")
        # RBF options
        func   = kw.get("function", "cubic")
        smooth = kw.get("smooth", 0.0)
        # Create tuple of input points
        V = tuple(self.get_values(arg, I) for arg in args)
        # Append reference values to input tuple
        Z = V + (self.get_values(col, I),)
        # Create a single RBF
        rbf = scirbf.Rbf(*Z, function=func, smooth=smooth)
        # Output
        return rbf

   # --- Griddata ---
    # Individual griddata generator
    def genr8_griddata_weights(self, args, *a, **kw):
        r"""Generate interpolation weights for :func:`griddata`

        :Call:
            >>> W = db.genr8_griddata_weights(arg, *a, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *a*: :class:`tuple`\ [:class:`np.ndarray`]
                Test values at which to interpolate
            *mask*: :class:`np.ndarray`\ [:class:`bool`]
                Mask of which database indices to consider
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Database indices to consider
            *method*: {``"linear"``} | ``"cubic"`` | ``"nearest"``
                Interpolation method; ``"cubic"`` only for 1D or 2D
            *rescale*: ``True`` | {``False``}
                Rescale input points to unit cube before interpolation
        :Outputs:
            *W*: :class:`np.ndarray`\ [:class:`float`]
                Interpolation weights; same size as test points *a*
        :Versions:
            * 2020-03-10 ``@ddalle``: Version 1.0
        """
        # Check for module
        if sciint is None:
            raise ImportError("No scipy.interpolate module")
        # Ensure list (len=2)
        if not isinstance(args, list):
            raise TypeError("Args must be a list, got '%s'" % type(args))
        elif len(args) == 0:
            raise ValueError("Arg list cannot be empty")
        # Number of args
        narg = len(args)
        # Check args
        for (j, arg) in enumerate(args):
            # Check string
            if not typeutils.isstr(arg):
                raise TypeError("Arg %i is not a string" % j)
        # Get method
        method = kw.get("method", "linear")
        # Other :func:`griddata` options
        rescale = kw.get("rescale", False)
        # Check it
        if method not in ["linear", "cubic", "nearest"]:
            # Invalid method
            raise ValueError("'method' must be either 'linear' or 'cubic'")
        # Check consistency
        if (method == "cubic") and (narg > 2):
            raise ValueError(
                "'cubic' method is for at most 2 args (got %i)" % narg)
        # Number of positional inputs
        na = len(a)
        # Check values
        if na < narg:
            raise TypeError(
                "At least %i positional args required (got %i)" % (narg, na))
        elif na > 3:
            raise TypeError(
                "At most %i positional args allowed (got %i)"
                % (narg + 1, na))
        # Get indices
        if na > narg:
            # Initialize with third arg
            I = a[narg]
        else:
            # Leave blank for Now
            I = None
        # Get *I* from kwargs
        mask = kw.get("I", kw.get("mask", None))
        # Prepare mask
        I = self.prep_mask(mask, args[0])
        # Get values of args from database
        x = np.vstack(tuple([self.get_values(arg, I)] for arg in args)).T
        # Get output values
        y = np.vstack(tuple([ai] for ai in a[:narg])).T
        # Length of input and output
        n = x.shape[0]
        nout = len(a[0])
        # Initialize weights
        W = np.zeros((nout, n))
        # Loop through evaluation points
        for k in range(n):
            # Artificial values
            kmode = np.eye(n)[k]
            # Calculate scattered interpolation weights
            W1 = sciint.griddata(x, kmode, y, method, rescale=rescale)
            W2 = sciint.griddata(x, kmode, y, "nearest", rescale=rescale)
            # Find any NaNs from extrapolation
            K = np.isnan(W1)
            # Replace NaNs with nearest value
            W1[K] = W2[K]
            # Save weights
            W[:,k] = W1
        # Output
        return W
  # >

  # ==================
  # Filtering
  # ==================
  # <
   # --- Repeats ---
    # Remove repeats
    def filter_repeats(self, args, cols=None, **kw):
        r"""Remove duplicate points or close neighbors

        :Call:
            >>> db.filter_repeats(args, cols=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *args*: :class:`list`\ [:class:`str`]
                List of columns names to match
            *cols*: {``None``} | :class:`list`\ [:class:`str`]
                Columns to filter (default is all *db.cols* with correct
                size and not in *args* and :class:`float` type)
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Subset of *db* to consider
            *function*: {``"mean"``} | **callable**
                Function to use for filtering
            *translators*: :class:`dict`\ [:class:`str`]
                Alternate names; *col* -> *trans[col]*
            *prefix*: :class:`str` | :class:`dict`
                Universal prefix or *col*-specific prefixes
            *suffix*: :class:`str` | :class:`dict`
                Universal suffix or *col*-specific suffixes
            *kw*: :class:`dict`
                Additional values to use for evaluation in :func:`find`
        :Versions:
            * 2020-05-05 ``@ddalle``: Version 1.0
        """
       # --- Column Lists ---
        # Check types
        if not isinstance(args, list):
            raise TypeError(
                "Invalid type '%s' for args, must be 'list'" % type(args))
        elif len(args) < 1:
            raise ValueError("Arg list is empty")
        # Check types of *args* entries
        for j, arg in enumerate(args):
            # Make sure it's a string
            if not typeutils.isstr(arg):
                raise TypeError(
                    "Arg %i has invalid type '%s', must be 'str'"
                    % (j, type(arg)))
        # Get first arg
        arg0 = args[0]
        # Get count
        n0 = len(self.get_all_values(arg0))
        # Process columns
        if cols is None:
            # Initialize list
            cols = []
            # Loop through columns
            for col in self.cols:
                # Check if an *arg*
                if col in args:
                    continue
                # Get data type
                dtype = self.get_col_dtype(col)
                # Check datatype
                if dtype is None or not dtype.startswith("float"):
                    continue
                # Get dimension
                ndim = self.get_col_prop(col, "Dimension")
                # Filter it
                if ndim is None or ndim != 1:
                    continue
                # Check size
                if self.get_all_values(col).size != n0:
                    continue
                # Otherwise, use this column
                cols.append(col)
       # --- Filtering Function ---
        # Get filtering function
        func = kw.get("func", "mean")
        # Filter it
        if func == "mean":
            # Take mean of data points
            fn = np.mean
        elif callable(func):
            # Function specified directly
            fn = func
        else:
            # Bad type
            raise TypeError("Filtering function not callable")
       # --- Options ---
        # Get translators
        trans = kw.pop("translators", {})
        prefix = kw.pop("prefix", None)
        suffix = kw.pop("suffix", None)
        # Overall mask
        mask = kw.get("mask")
        # Translator args
        tr_args = (trans, prefix, suffix)
        # Get overall tolerances
        tol = kw.get("tol", 1e-8)
        # Get tolerances for specific *args*
        tols = kw.get("tols", {})
       # --- Data ---
        # Divide into sweeps
        sweeps = self.genr8_sweeps(args, **kw)
        # Number of output points
        nx = len(sweeps)
        # Check for trivial filtering
        nmaxsweep = np.max(np.array([sweep.size for sweep in sweeps]))
        # Loop through *cols* first b/c original *arg* values
        # may affect filtering of *cols*
        for col in cols:
            # Translate column name
            colreg = self._translate_colname(col, *tr_args)
            # Check for trivial case
            if nmaxsweep == 1:
                # Get existing data
                V = self.get_values(col, mask)
            else:
                # Get data type
                dtype = self.get_col_dtype(col)
                # Initialize data
                V = np.zeros(nx, dtype=dtype)
                # Loop through sweeps
                for j, sweep in enumerate(sweeps):
                    # Get values for these points
                    v = self.get_values(col, sweep)
                    # Filter
                    V[j] = fn(v)
            # Check if new column
            if col != colreg:
                # Save new column
                self.save_col(colreg, V)
                # Get previous definition
                defn = self.get_defn(col)
                # Save a copy
                self.defns[colreg] = self._defncls(_warnmode=0, **defn)
            else:
                # Save new data in place of old data
                self[col] = V
        # Loop through *args*
        for arg in args:
            # Translate column name
            argreg = self._translate_colname(arg, *tr_args)
            # Check for trivial case
            if nmaxsweep == 1:
                # Get existing data
                V = self.get_values(arg, mask)
            else:
                # Get data type
                dtype = self.get_col_dtype(arg)
                # Initialize data
                V = np.zeros(nx, dtype=dtype)
                # Loop through sweeps
                for j, sweep in enumerate(sweeps):
                    # Get values for these points
                    v = self.get_values(arg, sweep)
                    # Filter
                    V[j] = fn(v)
            # Check if new column
            if arg != argreg:
                # Save new column
                self.save_col(argreg, V)
                # Get previous definition
                defn = self.get_defn(arg)
                # Save a copy
                self.defns[argreg] = self._defncls(_warnmode=0, **defn)
            else:
                # Save new data in place of old data
                self[arg] = V

    # Find duplicates
    def find_repeats(self, cols, **kw):
        r"""Find repeats based on list of columns

        :Call:
            >>> repeats = db.find_repeats(cols, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *cols*: :class:`list`\ [:class:`str`]
                List of columns names to match
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Subset of *db* to consider
            *tol*: {``1e-4``} | :class:`float` >= 0
                Default tolerance for all *args*
            *tols*: {``{}``} | :class:`dict`\ [:class:`float` >= 0]
                Dictionary of tolerances specific to arguments
            *kw*: :class:`dict`
                Additional values to use during evaluation
        :Outputs:
            *repeats*: :class:`list`\ [:class:`np.ndarray`]
                List of *db* indices of repeats; each *repeat* in
                *repeats* is an index of a case that matches for each
                *col* in *cols*
        :Versions:
            * 2021-09-10 ``@ddalle``: Version 1.0
        """
        # Set (force) option for db.find()
        kw["mapped"] = True
        # Get values for each *col*
        x = [self.get_all_values(col) for col in cols]
        # Search for duplicates
        Imap, _ = self.find(cols, *x, **kw)
        # Create set of first index of repeat
        anchors = set()
        # Initialize repeats
        repeats = []
        # Loop through *Imap* entries
        for imap in Imap:
            # Check if multiple
            if imap.size <= 1:
                continue
            # Check if already anchored
            if imap[0] in anchors:
                continue
            # Save repeate
            repeats.append(imap)
            # Save anchor
            anchors.add(imap[0])
        # Output
        return repeats
  # >

  # ==================
  # Data
  # ==================
  # <
   # --- Save/Add ---
   # --- Sort ---
    # Sort by list of columns
    def sort(self, cols=None):
        r"""Sort (ascending) using list of *cols*

        :Call:
            >>> db.sort(cols=None)
        :Inputs:
            *db*: :class:`DataKit`
                Data interface with response mechanisms
            *cols*: {``None``} | :class:`list`\ [:class:`str`]
                List of columns on which t sort, with highest sort
                priority to the first *col*, later *cols* used as
                tie-breakers
        :Versions:
            * 2021-09-17 ``@ddalle``: Version 1.0
        """
        # Default columns
        if cols is None:
            cols = list(self.cols)
        # First column
        col0 = cols[0]
        # Get value
        v0 = self.get_all_values(col0)
        # Size
        n0 = len(v0)
        # Get sorting order
        I = self.argsort(cols)
        # Loop through all columns
        for col in self.cols:
            # Get value
            v = self.get_all_values(col)
            # Check length
            if len(v) != n0:
                continue
            # Check type
            if isinstance(v, list):
                # Use a generator to reorder a list
                v = [v[i] for i in I]
            elif isinstance(v, np.ndarray):
                # Check dimension
                if v.ndim == 0:
                    # No sortable data
                    continue
                else:
                    # Sort on first axis
                    v = v[I]
            # Save sorted values
            self[col] = v
        
    # Sort by list of columns (get order)
    def argsort(self, cols=None):
        r"""Get (ascending) sort order using list of *cols*

        :Call:
            >>> I = db.argsort(cols=None)
        :Inputs:
            *db*: :class:`DataKit`
                Data interface with response mechanisms
            *cols*: {``None``} | :class:`list`\ [:class:`str`]
                List of columns on which t sort, with highest sort
                priority to the first *col*, later *cols* used as
                tie-breakers
        :Outputs:
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Ordering such that *db[cols[0]][I]* is ascending, etc.
        :Versions:
            * 2021-09-17 ``@ddalle``: Version 1.0
        """
        # Default columns
        if cols is None:
            cols = list(self.cols)
        # First column
        col0 = cols[0]
        # Get value
        v0 = self.get_all_values(col0)
        # Size
        n0 = len(v0)
        # Start args to :func:`np.lexsort`
        sort_args = []
        # Loop through columns
        for col in cols:
            # Get value
            v = self.get_all_values(col)
            # Check length
            if len(v) != n0:
                # Can't sort this arg
                continue
            # Check for 2D array
            if isinstance(v, np.ndarray) and v.ndim != 1:
                # Unable to sort matrices
                continue
            # Prepend to list (lexsort() prioritizes last arg)
            sort_args.insert(0, v)
        # Sort
        return np.lexsort(sort_args)

   # --- Copy/Link ---
    # Append data
    def append_data(self, dbsrc, cols=None, **kw):
        r"""Save one or more cols from another database

        .. note::

            This is the same as :func:`link_data` but with *append*
            defaulting to ``True``.

        :Call:
            >>> db.append_data(dbsrc, cols=None)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *dbsrc*: :class:`dict`
                Additional data container, not required to be a datakit
            *cols*: {``None``} | :class:`list`\ [:class:`str`]
                List of columns to link (or *dbsrc.cols*)
            *append*: {``True``} | ``False``
                Option to append data (or replace it)
            *prefix*: {``None``} | :class:`str`
                Prefix applied to *dbsrc* col when saved in *db*
            *suffix*: {``None``} | :class:`str`
                Prefix applied to *dbsrc* col when saved in *db*
        :Effects:
            *db.cols*: :class:`list`\ [:class:`str`]
                Appends each *col* in *cols* where not present
            *db[col]*: *dbsrc[col]*
                Reference to *dbsrc* data for each *col*
        :Versions:
            * 2021-09-10 ``@ddalle``: Version 1.0
        """
        # Set *append* to ``True``
        kw.setdefault("append", True)
        # Link data
        self.link_data(dbsrc, cols, **kw)
        
    # Link data
    def link_data(self, dbsrc, cols=None, **kw):
        r"""Save one or more cols from another database

        :Call:
            >>> db.link_data(dbsrc, cols=None)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *dbsrc*: :class:`dict`
                Additional data container, not required to be a datakit
            *cols*: {``None``} | :class:`list`\ [:class:`str`]
                List of columns to link (or *dbsrc.cols*)
            *append*: ``True`` | {``False``}
                Option to append data (or replace it)
            *prefix*: {``None``} | :class:`str`
                Prefix applied to *dbsrc* col when saved in *db*
            *suffix*: {``None``} | :class:`str`
                Prefix applied to *dbsrc* col when saved in *db*
        :Effects:
            *db.cols*: :class:`list`\ [:class:`str`]
                Appends each *col* in *cols* where not present
            *db[col]*: *dbsrc[col]*
                Reference to *dbsrc* data for each *col*
        :Versions:
            * 2019-12-06 ``@ddalle``: Version 1.0
            * 2021-09-10 ``@ddalle``: Version 1.1; *prefix* and *suffix*
        """
        # Check type of data set
        if not isinstance(dbsrc, dict):
            # Source must be a dictionary
            raise TypeError("Source data must be a dict")
        # Default columns
        if cols is None:
            # Check for explicit list
            if "cols" in dbsrc.__dict__:
                # Explicit list
                cols = dbsrc.cols
            else:
                # Get all keys
                cols = list(dbsrc.keys())
        # Check type of *cols*
        if not isinstance(cols, list):
            # Column list must be a list
            raise TypeError(
                "Column list must be a list, got '%s'"
                % cols.__class__.__name__)
        # Append option
        append = kw.get("append", False)
        # Prefix/suffix
        prefix = kw.get("prefix")
        suffix = kw.get("suffix")
        # Loop through columns
        for col in cols:
            # Check type
            if not typeutils.isstr(col):
                raise TypeError("Column names must be strings")
            # Check if data is present
            if col not in dbsrc:
                raise KeyError("No column '%s'" % col)
            # Candidate data
            v = dbsrc[col]
            # Add prefix and suffix to output column (col in *self*)
            if prefix:
                col1 = prefix + col
            else:
                col1 = col
            if suffix:
                col1 = col1 + suffix
            # Get data to save
            if append and col1 in self:
                # Get current values
                v0 = self[col1]
                # Check consistent types and combine values
                if (
                    isinstance(v, float)
                    and isinstance(v0, np.ndarray) and v0.ndim == 1
                ):
                    # Special case of mismatching types
                    v = np.append(v0, v)
                elif not (isinstance(v0, type(v)) or isinstance(v, type(v0))):
                    # No way to combine
                    sys.stderr.write(
                        "Cannot combine old and new values for col '%s'\n"
                        % col)
                    sys.stderr.flush()
                elif isinstance(v, list):
                    # Combine lists
                    v = v0 + v
                elif isinstance(v, float):
                    # Make an array
                    v = np.array([v0, v])
                elif isinstance(v, np.ndarray):
                    # Check dimensions
                    if v0.ndim == 1 and v.ndim == 0:
                        # Append scalar
                        v = np.append(v0, v)
                    elif v0.ndim != v.ndim:
                        # Mismatching sizes
                        sys.stderr.write(
                            "Cannot combine %iD and %iD arrays for '%s'\n"
                            % (v0.ndim, v.ndim, col))
                        sys.stderr.flush()
                    elif v0.ndim == 1:
                        # Stack
                        v = np.hstack((v0, v))
                    elif v0.ndim == 2:
                        # Stack vertically
                        v = np.vstack((v0, v))
                    else:
                        # No way to combine 3D matrices
                        sys.stderr.write(
                            "Cannot combine %iD arrays for '%s'\n"
                            % (v.ndim, col))
                        sys.stderr.flush()
                else:
                    # No general fallback combination
                    sys.stderr.write(
                        "Cannot combine old and new values for col '%s'\n"
                        % col)
                    sys.stderr.flush()
            # Save the data
            self.save_col(col1, v)

   # --- Access ---
    # Look up a generic key
    def get_col(self, k=None, defnames=[], **kw):
        r"""Process a key name, using an ordered list of defaults

        :Call:
            >>> col = db.get_key(k=None, defnames=[], **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *k*: {``None``} | :class:`str`
                User-specified col name; if ``None``, automatic value
            *defnamess*: :class:`list`
                List of applicable default names for the col
            *title*: {``"lookup"``} | :class:`str`
                Key title to use in any error messages
            *error*: {``True``} | ``False``
                Raise an exception if no col is found
        :Outputs:
            *col*: *k* | *defnamess[0]* | *defnamess[1]* | ...
                Name of lookup key in *db.cols*
        :Versions:
            * 2018-06-22 ``@ddalle``: Version 1.0
        """
        # Get default
        if (k is not None):
            # Check if it's present
            if k not in self.cols:
                # Error option
                if kw.get("error", True):
                    # Get title for error message, e.g. "angle of attack"
                    ttl = kw.get("title", "lookup")
                    # Error message
                    raise KeyError("No %s key found" % ttl)
                else:
                    # If no error, return ``None``
                    return defnames[0]
            # Otherwise, done
            return k
        # Check list
        if not isinstance(defnames, list):
            raise TypeError("Default col names must be list")
        # Loop through defaults
        for col in defnames:
            # Check if it's present
            if col in self.cols:
                return col
        # If this point is reached, no default found
        if not kw.get("error", True):
            # No error, as specified by user
            return defnames[0]
        # Get title for error message, e.g. "angle of attack"
        ttl = kw.get("title", "lookup")
        # Error message
        raise KeyError("No %s key found" % ttl)

   # --- Independent Key Values ---
    # Get the value of an independent variable if possible
    def get_xvals(self, col, I=None, **kw):
        r"""Get values of specified column, which may need conversion

        This function can be used to calculate independent variables
        (*xvars*) that are derived from extant data columns.  For
        example if columns *alpha* and *beta* (for angle of attack and
        angle of sideslip, respectively) are present and the user wants
        to get the total angle of attack *aoap*, this function will
        attempt to use ``db.response_arg_converters["aoap"]`` to convert
        available *alpha* and *beta* data.

        :Call:
            >>> V = db.get_xvals(col, I=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to access
            *I*: ``None`` | :class:`np.ndarray` | :class:`int`
                Subset indices or single index
            *kw*: :class:`dict`
                Dictionary of values in place of *db* (e.g. *kw[col]*
                instead of *db[col]*)
            *IndexKW*: ``True`` | {``False``}
                Option to use *kw[col][I]* instead of just *kw[col]*
        :Outputs:
            *V*: :class:`np.ndarray` | :class:`float`
                Array of values or scalar for column *col*
        :Versions:
            * 2019-03-12 ``@ddalle``: Version 1.0
            * 2019-12-26 ``@ddalle``: From :mod:`tnakit.db.db1`
        """
        # Option for processing keywrods
        qkw = kw.pop("IndexKW", False)
        # Check for direct membership
        if col in kw:
            # Get values from inputs
            V = kw[col]
            # Checking for quick output
            if not qkw:
                return V
        elif col in self:
            # Get all values from column
            V = self[col]
        else:
            # Get converter
            f = self.get_response_arg_converter(col)
            # Check for converter
            if f is None:
                raise ValueError("No converter for col '%s'" % col)
            elif not callable(f):
                raise TypeError("Converter for col '%s' not callable" % col)
            # Create a dictionary of values
            X = dict(self, **kw)
            # Attempt to convert
            try:
                # Use entire dictionary as inputs
                V = f(**X)
            except Exception:
                raise ValueError("Conversion function for '%s' failed" % col)
        # Check for indexing
        if I is None:
            # No subset
            return V
        elif typeutils.isarray(V):
            # Apply subset
            return V[I]
        else:
            # Not subsettable
            return V

    # Get independent variable from eval inputs
    def get_xvals_eval(self, k, *a, **kw):
        r"""Return values of a column from inputs to :func:`__call__`

        For example, this can be used to derive the total angle of
        attack from inputs to an evaluation call to *CN* when it is a
        function of *mach*, *alpha*, and *beta*.  This method attempts
        to use :func:`db.response_arg_converters`.

        :Call:
            >>> V = db.get_xvals_eval(k, *a, **kw)
            >>> V = db.get_xvals_eval(k, coeff, x1, x2, ..., k3=x3)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *k*: :class:`str`
                Name of key to calculate
            *col*: :class:`str`
                Name of output data column
            *x1*: :class:`float` | :class:`np.ndarray`
                Value(s) of first argument
            *x2*: :class:`float` | :class:`np.ndarray`
                Value(s) of second argument, if applicable
            *k3*: :class:`str`
                Name of third argument or optional variant
            *x3*: :class:`float` | :class:`np.ndarray`
                Value(s) of argument *k3*, if applicable
        :Outputs:
            *V*: :class:`np.ndarray`
                Values of key *k* from conditions in *a* and *kw*
        :Versions:
            * 2019-03-12 ``@ddalle``: Version 1.0
            * 2019-12-26 ``@ddalle``: From :mod:`tnakit`
        """
        # Process coefficient
        X = self.get_arg_value_dict(*a, **kw)
        # Check if key is present
        if k in X:
            # Return values
            return X[k]
        else:
            # Get dictionary of converters
            converters = getattr(self, "response_arg_converters", {})
            # Check for membership
            if k not in converters:
                raise ValueError(
                    "Could not interpret xvar '%s'" % k)
            # Get converter
            f = converters[k]
            # Check class
            if not callable(f):
                raise TypeError("Converter for '%s' is not callable" % k)
            # Attempt to convert
            try:
                # Use entire dictionary as inputs
                V = f(**X)
            except Exception:
                raise ValueError("Conversion function for '%s' failed" % k)
            # Output
            return V

   # --- Dependent Key Values ---
    # Get exact values
    def get_yvals_exact(self, col, I=None, **kw):
        r"""Get exact values of a data column

        :Call:
            >>> V = db.get_yvals_exact(col, I=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of column to access
            *I*: {``None``} | :class:`np.ndarray`\ [:class:`int`]
                Database indices
        :Versions:
            * 2019-03-13 ``@ddalle``: Version 1.0
            * 2019-12-26 ``@ddalle``: From :mod:`tnakit`
        """
        # Check for direct membership
        if col in self:
            # Get all values from column
            V = self[col]
            # Check for indexing
            if I is None:
                # No subset
                return V
            else:
                # Apply subset
                return V[I]
        else:
            # Get evaluation type
            meth = self.get_response_method(col)
            # Only allow "function" type
            if meth != "function":
                raise ValueError(
                    ("Cannot evaluate exact values for '%s', " % col) +
                    ("which has method '%s'" % meth))
            # Get args
            args = self.get_response_args(col)
            # Create inputs
            a = tuple([self.get_xvals(k, I, **kw) for k in args])
            # Evaluate
            V = self.__call__(col, *a)
            # Output
            return V

   # --- Subsets ---
    # Attempt to get all values of an argument
    def get_all_values(self, col):
        r"""Attempt to get all values of a specified argument

        This will use *db.response_arg_converters* if possible.

        :Call:
            >>> V = db.get_all_values(col)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column
        :Outputs:
            *V*: ``None`` | :class:`np.ndarray`\ [:class:`float`]
                *db[col]* if available, otherwise an attempt to apply
                *db.response_arg_converters[col]*
        :Versions:
            * 2019-03-11 ``@ddalle``: Version 1.0
            * 2019-12-18 ``@ddalle``: Ported from :mod:`tnakit`
        """
        # Check if present
        if col in self:
            # Get values
            return self[col]
        # Otherwise check for evaluation argument
        arg_converters = self.__dict__.get("response_arg_converters", {})
        # Check if there's a converter
        if col not in arg_converters:
            return None
        # Get converter
        f = arg_converters.get(col)
        # Check if there's a converter
        if f is None:
            # No converter
            return
        elif not callable(f):
            # Not callable
            raise TypeError("Converter for col '%s' is not callable" % col)
        # Attempt to apply it
        try:
            # Call in keyword-only mode
            V = f(**self)
            # Return values
            return V
        except Exception:
            # Failed
            return None

    # Attempt to get values of an argument or column, with mask
    def get_values(self, col, mask=None):
        r"""Attempt to get all or some values of a specified column

        This will use *db.response_arg_converters* if possible.

        :Call:
            >>> V = db.get_values(col)
            >>> V = db.get_values(col, mask=None)
            >>> V = db.get_values(col, mask_index)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of evaluation argument
            *I*: :class:`np.ndarray`\ [:class:`int` | :class:`bool`]
                Optional subset of *db* indices to access
        :Outputs:
            *V*: ``None`` | :class:`np.ndarray`\ [:class:`float`]
                *db[col]* if available, otherwise an attempt to apply
                *db.response_arg_converters[col]*
        :Versions:
            * 2020-02-21 ``@ddalle``: Version 1.0
        """
        # Get all values
        V = self.get_all_values(col)
        # Check for empty result
        if V is None:
            return
        # Check for mask
        if mask is None:
            # No mask
            return V
        # Otherwise check validity of mask
        I = self.prep_mask(mask, V=V)
        # Dimension
        if isinstance(V, list):
            # Index directly
            # (Lists always 1D)
            if I.ndim == 0:
                # Scalar mask
                return V[I]
            elif len(I) > 0 and isinstance(I[0], bool):
                # Apply boolean mask
                return [v for i, v in enumerate(V) if mask[i]]
            else:
                # Apply index mask
                return [V[i] for i in I]
        # Get array dimension
        ndim = V.ndim
        # Create slice that looks up last column
        J = tuple(slice(None) for j in range(ndim-1)) +  (I,)
        # Apply mask to last dimension
        return V.__getitem__(J)

    # Apply a mask to all columns
    def apply_mask(self, mask, cols=None):
        r"""Apply a mask to one or more *cols*

        :Call:
            >>> db.apply_mask(mask, cols=None)
            >>> db.apply_mask(mask_index, cols=None)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *mask*: {``None``} | :class:`np.ndarray`\ [:class:`bool`]
                Logical mask of ``True`` / ``False`` values
            *mask_index*: :class:`np.ndarray`\ [:class:`int`]
                Indices of values to consider
            *cols*: {``None``} | :class:`list`\ [:class:`str`]
                List of columns to subset (default is all)
        :Effects:
            *db[col]*: :class:`list` | :class:`np.ndarray`
                Subset *db[col][mask]* or similar
        :Versions:
            * 2021-09-10 ``@ddalle``: Version 1.0
        """
        # Default list of columns
        if cols is None:
            cols = self.cols
        # Loop through columns
        for col in cols:
            # Check validity of mask
            if not self.check_mask(mask, col):
                # Skip this column
                continue
            # Get values according to *mask*
            v = self.get_values(col, mask)
            # Save subsetted values
            self[col] = v

    # Apply a mask to all columns
    def remove_mask(self, mask, cols=None):
        r"""Remove cases in a mask for one or more *cols*

        This function is the opposite of :func:`apply_mask`

        :Call:
            >>> db.remove_mask(mask, cols=None)
            >>> db.remove_mask(mask_index, cols=None)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *mask*: {``None``} | :class:`np.ndarray`\ [:class:`bool`]
                Logical mask of ``True`` / ``False`` values
            *mask_index*: :class:`np.ndarray`\ [:class:`int`]
                Indices of values to delete
            *cols*: {``None``} | :class:`list`\ [:class:`str`]
                List of columns to subset (default is all)
        :Effects:
            *db[col]*: :class:`list` | :class:`np.ndarray`
                Subset *db[col][mask]* or similar
        :Versions:
            * 2021-09-10 ``@ddalle``: Version 1.0
        """
        # Check for null action
        if mask is None:
            return
        if len(mask) == 0:
            return
        if not np.any(mask):
            return
        # Default list of columns
        if cols is None:
            cols = self.cols
        # Get size of first *col*
        n = len(self[cols[0]])
        # Create mask of what to *keep*
        pmask = np.full(n, True)
        # Cases to remove
        pmask[mask] = False
        # Apply tyat
        self.apply_mask(pmask ,cols)

   # --- Mask ---
    # Prepare mask
    def prep_mask(self, mask, col=None, V=None):
        r"""Prepare logical or index mask

        :Call:
            >>> I = db.prep_mask(mask, col=None, V=None)
            >>> I = db.prep_mask(mask_index, col=None, V=None)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *mask*: {``None``} | :class:`np.ndarray`\ [:class:`bool`]
                Logical mask of ``True`` / ``False`` values
            *mask_index*: :class:`np.ndarray`\ [:class:`int`]
                Indices of *db[col]* to consider
            *col*: {``None``} | :class:`str`
                Reference column to use for size checks
            *V*: {``None``} | :class:`np.ndarray`
                Array of values to test shape/values of *mask*
        :Outputs:
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Indices of *db[col]* to consider
        :Versions:
            * 2020-03-09 ``@ddalle``: Version 1.0
        """
        # Get data so size can be determined
        if V is None:
            # Check for column
            if col is None:
                raise ValueError("Either column or array must be specified")
            # Get all values
            V = self.get_all_values(col)
        # Assert validity of mask
        self.assert_mask(mask, V=V)
        # Length of reference array
        if isinstance(V, list):
            # Just use length for lists
            n0 = len(V)
        else:
            # Use last dimension
            n0 = V.shape[-1]
        # Ensure array
        if isinstance(mask, (list, int)):
            mask = np.array(mask)
        # Get data type
        if mask is not None:
            dtype = mask.dtype.name
        # Filter mask
        if mask is None:
            # Create indices
            mask_index = np.arange(n0)
        elif dtype.startswith("bool"):
            # Get indices
            mask_index = np.where(mask)[0]
        elif dtype.startswith("int") or dtype.startswith("uint"):
            # Already indices
            mask_index = mask
        else:
            # Bad type
            # Note: should not be reachable
            raise TypeError("Mask must have dtype 'bool' or 'int'")
        # Output
        return mask_index

    # Check if mask
    def check_mask(self, mask, col=None, V=None):
        r"""Check if *mask* is a valid index/bool mask

        :Call:
            >>> q = db.check_mask(mask, col=None, V=None)
            >>> q = db.check_mask(mask_index, col=None, V=None)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *mask*: {``None``} | :class:`np.ndarray`\ [:class:`bool`]
                Logical mask of ``True`` / ``False`` values
            *mask_index*: :class:`np.ndarray`\ [:class:`int`]
                Indices of values to consider
            *col*: {``None``} | :class:`str`
                Column name to use to create default *V*
            *V*: {``None``} | :class:`np.ndarray`
                Array of values to test shape/values of *mask*
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not *mask* is a valid mask
        :Versions:
            * 2020-04-21 ``@ddalle``: Version 1.0
        """
        # Check for empty mask
        if mask is None:
            return True
        # Get values
        if (V is None) and (col is not None):
            # Use all values
            V = self.get_all_values(col)
        # Convert list to array
        if isinstance(mask, (list, int)):
            # Create an array instead of a list
            mask = np.array(mask)
        # Check mask type and dimension
        if not isinstance(mask, np.ndarray):
            # Must be array
            return False
        elif mask.ndim > 1:
            # Bad dimension
            return False
        elif mask.size == 0:
            # Empty mask (``None`` is correct empty mask)
            return False
        # Check dimensions
        if V is not None:
            # Get size
            if isinstance(V, list):
                # Special case for list of strings
                n = len(V)
            else:
                # Use last dimension
                n = V.shape[-1]
        # Get data type
        dtype = mask.dtype.name
        # Check data type
        if dtype.startswith("uint") or dtype.startswith("int"):
            # Check values
            if V is not None:
                # Check values
                if np.min(mask) < -n:
                    return False
                elif np.max(mask) >= n:
                    return False
            # Ints are valid masks
            return True
        elif dtype.startswith("bool"):
            # Check size
            if V is not None:
                # Check length
                if mask.size != n:
                    return False
            # Boolean masks are valid if dimension matches
            return True
        else:
            # Invalid data type
            return False

    # Ensure mask
    def assert_mask(self, mask, col=None, V=None):
        r"""Make sure that *mask* is a valid index/bool mask

        :Call:
            >>> db.assert_mask(mask, col=None, V=None)
            >>> db.assert_mask(mask_index, col=None, V=None)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *mask*: ``None`` | :class:`np.ndarray`\ [:class:`bool`]
                Logical mask of ``True`` / ``False`` values
            *mask_index*: :class:`np.ndarray`\ [:class:`int`]
                Indices of values to consider
            *col*: {``None``} | :class:`str`
                Column name to use to create default *V*
            *V*: {``None``} | :class:`np.ndarray`
                Array of values to test shape/values of *mask*
        :Versions:
            * 2020-04-21 ``@ddalle``: Version 1.0
        """
        # Check for empty mask
        if mask is None:
            return
        # Get values
        if (V is None) and (col is not None):
            # Use all values
            V = self.get_all_values(col)
        # Convert list to array
        if isinstance(mask, (list, int, np.int32, np.int64)):
            # Create an array instead of a list
            mask = np.array(mask)
        # Check mask type and dimension
        if not isinstance(mask, np.ndarray):
            # Must be array
            raise TypeError("Index mask must be NumPy array")
        elif mask.ndim > 1:
            # Bad dimension
            raise IndexError(
                "%iD index mask; must be scalar or 1D array" % mask.ndim)
        elif mask.size == 0:
            # Empty mask (``None`` is correct empty mask)
            raise ValueError("Index mask must not be empty")
        # Check dimensions
        if V is not None:
            # Get size
            if isinstance(V, list):
                # Special case for list of strings
                n = len(V)
            else:
                # Use last dimension
                n = V.shape[-1]
        # Get data type
        dtype = mask.dtype.name
        # Check data type
        if dtype.startswith("uint") or dtype.startswith("int"):
            # Check values
            if V is not None:
                # Check values
                if np.min(mask) < -n:
                    raise ValueError(
                        "Index %i is outside of range for array of length %i"
                        % (np.min(mask), n))
                elif np.max(mask) >= n:
                    raise ValueError(
                        "Index %i is outside of range for array of length %i"
                        % (np.max(mask), n))
        elif dtype.startswith("bool"):
            # Check size
            if V is not None:
                # Check length
                if mask.size != n:
                    raise ValueError(
                        "Boolean mask (%i) and array (%i) have different sizes"
                        % (mask.size, n))
        else:
            # Invalid data type
            raise TypeError("Invalid mask data type '%s'" % dtype)

   # --- Sweeps ---
    # Divide into sweeps
    def genr8_sweeps(self, args, **kw):
        r"""Divide data into sweeps with constant values of some *cols*

        :Call:
            >>> sweeps = db.genr8_sweeps(args, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *args*: :class:`list`\ [:class:`str`]
                List of columns names to match
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Subset of *db* to consider
            *tol*: {``1e-4``} | :class:`float` >= 0
                Default tolerance for all *args*
            *tols*: {``{}``} | :class:`dict`\ [:class:`float` >= 0]
                Dictionary of tolerances specific to arguments
            *kw*: :class:`dict`
                Additional values to use during evaluation
        :Outputs:
            *sweeps*: :class:`list`\ [:class:`np.ndarray`]
                Indices of entries with constant (within *tol*) values
                of each *arg*
        :Versions:
            * 2020-05-06 ``@ddalle``: Version 1.0
        """
       # --- Column Lists ---
        # Check arg list
        if not isinstance(args, list):
            raise TypeError(
                "Invalid type '%s' for args, must be 'list'" % type(args))
        elif len(args) < 1:
            raise ValueError("Arg list is empty")
        # Check types of *args* entries
        for j, arg in enumerate(args):
            # Make sure it's a string
            if not typeutils.isstr(arg):
                raise TypeError(
                    "Arg %i has invalid type '%s', must be 'str'"
                    % (j, type(arg)))
       # --- Options ---
        # Tolerances
        tol = kw.pop("tol", 1e-8)
        tols = kw.pop("tols", {})
       # --- Mask Prep ---
        # Initialize sweeps
        mask = kw.pop("mask", kw.pop("I", None))
        # Translate into indices
        mask_index = self.prep_mask(mask, args[0])
       # --- Search ---
        # Initialize sweeps [array[int]]
        sweeps_parent = [mask_index]
        # Loop through args
        for k, arg in enumerate(args):
            # Get tolerance
            ktol = tols.get(arg, tol)
            # Initialize new sweeps
            sweeps = []
            # Options to :func:`find`
            kw_k = dict(tol=ktol, mapped=True, **kw)
            # Loop through sweeps from previous level
            for j, J in enumerate(sweeps_parent):
                # Get unique values for this arg (within tolerance)
                V = self.genr8_bkpts(arg, nmin=0, mask=J, tol=ktol)
                # Divide into sweeps
                sweeps_j, _ = self.find([arg], V, mask=J, **kw_k)
                # Save these sweeps
                sweeps.extend(sweeps_j)
            # Save current sweeps
            sweeps_parent = sweeps
        # Output
        return sweeps

   # --- Search ---
    # Find matches
    def find(self, args, *a, **kw):
        r"""Find cases that match a condition [within a tolerance]

        :Call:
            >>> I, J = db.find(args, *a, **kw)
            >>> Imap, J = db.find(args, *a, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data container
            *args*: :class:`list`\ [:class:`str`]
                List of columns names to match
            *a*: :class:`tuple`\ [:class:`float`]
                Values of the arguments
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Subset of *db* to consider
            *tol*: {``1e-4``} | :class:`float` >= 0
                Default tolerance for all *args*
            *tols*: {``{}``} | :class:`dict`\ [:class:`float` >= 0]
                Dictionary of tolerances specific to arguments
            *once*: ``True`` | {``False``}
                Option to find max of one *db* index per test point
            *mapped*: ``True`` | {``False``}
                Option to switch output to *Imap* (overrides *once*)
            *kw*: :class:`dict`
                Additional values to use during evaluation
        :Outputs:
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Indices of cases in *db* that match conditions
            *J*: :class:`np.ndarray`\ [:class:`int`]
                Indices of (*a*, *kw*) that have a match in *db*
            *Imap*: :class:`list`\ [:class:`np.ndarray`]
                List of *db* indices for each test point in *J*
        :Versions:
            * 2019-03-11 ``@ddalle``: Version 1.0
            * 2019-12-26 ``@ddalle``: From :func:`DBCoeff.FindMatches`
            * 2020-02-20 ``@ddalle``: Added *mask*, *once* kwargs
        """
       # --- Input Checks ---
        # Find a valid argument
        for arg in args:
            # Attempt to either access or convert it
            V = self.get_all_values(arg)
            # Check if it was processed
            if V is not None:
                # Found at least one valid argument
                break
        else:
            # Loop completed; nothing found
            raise ValueError(
                "Cannot find matches for argument list %s" % args)
        # Mask
        mask = kw.pop("mask", None)
        # Overall tolerance default
        tol = kw.pop("tol", 1e-4)
        # Specific tolerances
        tols = kw.pop("tols", {})
        # Option for unique matches
        once = kw.pop("once", False)
        # Option for mapped matches
        mapped = kw.pop("mapped", False)
        # Number of values
        n0 = len(V)
       # --- Mask Prep ---
        # Get mask
        mask_index = self.prep_mask(mask, arg)
        # Update test size
        n = mask_index.size
       # --- Argument values ---
        # Initialize lookup point
        x = []
        # Loop through arguments
        for i, col in enumerate(args):
            # Get value
            xi = self.get_arg_value(i, col, *a, **kw)
            # Save it
            x.append(np.asarray(xi))
        # Normalize arguments
        X, dims = self.normalize_args(x, True)
        # Number of test points
        nx = np.prod(dims)
       # --- Checks ---
        # Initialize tests for database indices (set to ``False``)
        MI = np.full(n, False)
        # Initialize tests for input data indices (set to ``False``)
        MJ = np.full(nx, False)
        # Initialize maps if needed
        if mapped:
            Imap = []
        # Loop through entries
        for i in range(nx):
            # Initialize tests for this point (set to ``True``)
            Mi = np.full(n, True)
            # Loop through arguments
            for j, k in enumerate(args):
                # Get array of database values
                Xk = self.get_all_values(k)
                # Check if present
                if (k is None) or (Xk is None):
                    continue
                # Check size
                if len(Xk) != n0:
                    raise ValueError(
                        ("Parameter '%s' has size %i, " % (k, len(Xk))) +
                        ("expecting %i" % n))
                # Apply mask
                if mask is not None:
                    Xk = self.get_values(k, mask)
                # Get input test value
                xi = X[j][i]
                # Get tolerance for this key
                xtol = tols.get(k, tol)
                # Check match/approx
                if isinstance(Xk, list):
                    # Convert to array
                    Xk = np.asarray(Xk)
                # Check for match
                if Xk.dtype.name.startswith("str"):
                    # Exact match for strings
                    Mi = np.logical_and(Mi, Xk == xi)
                else:
                    # Use a tolerance
                    Mi = np.logical_and(Mi, np.abs(Xk-xi) <= xtol)
            # Check if any cases
            found = np.any(Mi)
            # Got to next test point if no match
            if not found:
                # Save status
                MJ[i] = found
                continue
            # Check reporting method
            if mapped:
                # Save test-point status (no uniqueness check)
                MJ[i] = found
                # Find matches
                I = np.where(Mi)[0]
                # Invert mask if needed
                if mask is not None:
                    I = mask_index[I]
                # Append to map
                Imap.append(I)
            elif once:
                # Check for uniqueness
                M2 = np.logical_and(np.logical_not(MI), Mi)
                # Check that
                found = np.any(M2)
                # Save status
                MJ[i] = found
                # Exit if not found (match but previously used)
                if not found:
                    continue
                # Select first not-previously-used match
                j2 = np.where(M2)[0][0]
                # Save it
                MI[j2] = True
            else:
                # Save test-point status (no uniqueness check)
                MJ[i] = found
                # Combine point constraints (*Mi* multiple matches)
                MI = np.logical_or(MI, Mi)
        # Convert test point status to indices
        J = np.where(MJ)[0]
        # Convert database point mask to indices
        if mapped:
            # Output map and test point index array
            return Imap, J
        else:
            # Convert masks to indices
            I = np.where(MI)[0]
            # Invert mask if needed
            if mask is not None:
                I = mask_index[I]
            # Return combined set of matches
            return I, J

    # Find matches from a target
    def match(self, dbt, maskt=None, cols=None, **kw):
        r"""Find cases with matching values of specified list of cols

        :Call:
            >>> I, J = db.match(dbt, maskt, cols=None, **kw)
            >>> Imap, J = db.match(dbt, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data kit with response surfaces
            *dbt*: :class:`dict` | :class:`DataKit`
                Target data set
            *maskt*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Subset of *dbt* to consider
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Subset of *db* to consider
            *cols*: {``None``} | :class:`np.ndarray`\ [:class:`int`]
                List of cols to compare (default all *db* float cols)
            *tol*: {``1e-4``} | :class:`float` >= 0
                Default tolerance for all *args*
            *tols*: {``{}``} | :class:`dict`\ [:class:`float` >= 0]
                Dictionary of tolerances specific to arguments
            *once*: ``True`` | {``False``}
                Option to find max of one *db* index per test point
            *mapped*: ``True`` | {``False``}
                Option to switch output to *Imap* (overrides *once*)
            *kw*: :class:`dict`
                Additional values to use during evaluation
        :Outputs:
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Indices of cases in *db* that have a match in *dbt*
            *J*: :class:`np.ndarray`\ [:class:`int`]
                Indices of cases in *dbt* that have a match in *db*
            *Imap*: :class:`list`\ [:class:`np.ndarray`]
                List of *db* indices for each test point in *J*
        :Versions:
            * 2020-02-20 ``@ddalle``: Version 1.0
            * 2020-03-06 ``@ddalle``: Name from :func:`find_pairwise`
        """
        # Check types
        if not isinstance(dbt, dict):
            raise TypeError("Target database is not a DataKit")
        # Default columns
        if cols is None:
            # Take all columns with a "float" type
            cols = [col for col in self.cols
                if self.get_col_dtype(col).startswith("float")
            ]
        # Check *cols* type
        if not isinstance(cols, list):
            raise TypeError(
                "Column list must be 'list', got '%s'" % type(cols).__name__)
        # Check for nontrivial cols
        if len(cols) == 0:
            raise ValueError("Empty column list")
        # Check mask type
        if maskt is None:
            # Ok
            pass
        elif not isinstance(maskt, np.ndarray):
            # Bad type
            raise TypeError(
                "Target mask must be 'ndarray', got '%s'"
                % type(maskt).__name__)
        elif maskt.size == 0:
            # Empty mask
            raise IndexError("Target index mask cannot be empty")
        elif maskt.ndim != 1:
            # Dimension error
            raise IndexError("Target index mask must be one-dimensional array")
        # Filter mask
        if maskt is None:
            # No inversion
            pass
        elif maskt.dtype.name == "bool":
            # Get indices
            maskt_index = np.where(maskt)[0]
        elif maskt.dtype.name.startswith("int"):
            # Convert to indices
            maskt_index = maskt
        else:
            # Bad type
            raise TypeError("Target mask must have dtype 'bool' or 'int'")
        # Create list or args and their values to :func:`find`
        args = []
        argvals = []
        # Check mode for data-kit (versus generic dict) target
        isdatakit = isinstance(dbt, DataKit)
        # Loop through columns
        for col in cols:
            # Check type
            if not typeutils.isstr(col):
                raise TypeError(
                    "Col name must be 'str', got '%s'" % type(col).__name__)
            # Get *dbt* values
            if isdatakit:
                # Get value; use converters if necessary
                V = dbt.get_all_values(col)
            else:
                # Get value from a dict
                V = dbt.get(col)
            # Ensure array
            if V is None:
                # No match
                continue
            elif isinstance(V, (list, np.ndarray)):
                # Check size
                if len(V) == 0:
                    continue
                # Check data type
                if typeutils.isstr(V[0]):
                    # No strings
                    continue
                # If list, convert
                if isinstance(V, list):
                    # Force array
                    V = np.asarray(V)
                # Apply mask
                if maskt is not None:
                    V = V[maskt]
            elif not isinstance(V, (int, float, complex)):
                # Non-numeric type
                continue
            # Save to arg list
            args.append(col)
            # Save value
            argvals.append(V)
        # Find matches in *db* based on args
        I, J = self.find(args, *argvals, **kw)
        # Check for mask
        if maskt is not None:
            # Invert mask
            J = maskt_index[J]
        # Output
        return I, J

   # --- Statistics ---
    # Get coverage
    def est_cov_interval(self, dbt, col, mask=None, cov=0.95, **kw):
        r"""Calculate Student's t-distribution confidence region

        If the nominal application of the Student's t-distribution fails
        to cover a high enough fraction of the data, the bounds are
        extended until the data is covered.

        :Call:
            >>> a, b = db.est_cov_interval(dbt, col, mask, cov, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data kit with response surfaces
            *dbt*: :class:`dict` | :class:`DataKit`
                Target data set
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Subset of *db* to consider
            *maskt*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Subset of *dbt* to consider
            *cov*: {``0.95``} | 0 < :class:`float` < 1
                Coverage percentage
            *cdf*, *CoverageCDF*: {*cov*} | 0 < :class:`float` < 1
                CDF if no extra coverage needed
            *osig*, *OutlierSigma*: {``1.5*ksig``} | :class:`float`
                Multiple of standard deviation to identify outliers;
                default is 150% of the nominal coverage calculated using
                t-distribution
            *searchcols*: {``None``} | :class:`list`\ [:class:`str`]
                List of cols to use for finding matches; default is all
                :class:`float` cols of *db*
            *tol*: {``1e-8``} | :class:`float`
                Default tolerance for matching conditions
            *tols*: :class:`dict`\ [:class:`float`]
                Dict of tolerances for specific columns during search
        :Outputs:
            *a*: :class:`float`
                Lower bound of coverage interval
            *b*: :class:`float`
                Upper bound of coverage intervalregion
        :Versions:
            * 2018-09-28 ``@ddalle``: Version 1.0
            * 2020-02-21 ``@ddalle``: Rewritten from :mod:`cape.attdb.fm`
        """
        # Process search kwargs
        kw_find = {
            "mask": mask,
            "maskt": kw.pop("maskt", None),
            "once": True,
            "cols": kw.pop("searchcols", None),
            "tol": kw.get("tol", 1e-8),
            "tols": kw.get("tols", {}),
        }
        # Find indices of matches
        I, J = self.match(dbt, **kw_find)
        # Check for empty
        if I.size == 0:
            raise ValueError("No matches between databases")
        # Get values from this database
        V1 = self.get_values(col, I)
        # Get values from target database
        if isinstance(dbt, DataKit):
            # Get values with converters
            V2 = dbt.get_values(col, J)
        else:
            # Get values from dict
            V2 = dbt[col][J]
        # Deltas (signed)
        dV = V2 - V1
        # Calculate interval
        return statutils.get_cov_interval(dV, cov, **kw)

    # Get coverage
    def est_range(self, dbt, col, mask=None, cov=0.95, **kw):
        r"""Calculate Student's t-distribution confidence range

        If the nominal application of the Student's t-distribution fails
        to cover a high enough fraction of the data, the bounds are
        extended until the data is covered.

        :Call:
            >>> r = db.est_range(dbt, col, mask, cov, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Data kit with response surfaces
            *dbt*: :class:`dict` | :class:`DataKit`
                Target data set
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Subset of *db* to consider
            *maskt*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Subset of *dbt* to consider
            *cov*: {``0.95``} | 0 < :class:`float` < 1
                Coverage percentage
            *cdf*, *CoverageCDF*: {*cov*} | 0 < :class:`float` < 1
                CDF if no extra coverage needed
            *osig*, *OutlierSigma*: {``1.5*ksig``} | :class:`float`
                Multiple of standard deviation to identify outliers;
                default is 150% of the nominal coverage calculated using
                t-distribution
            *searchcols*: {``None``} | :class:`list`\ [:class:`str`]
                List of cols to use for finding matches; default is all
                :class:`float` cols of *db*
            *tol*: {``1e-8``} | :class:`float`
                Default tolerance for matching conditions
            *tols*: :class:`dict`\ [:class:`float`]
                Dict of tolerances for specific columns during search
        :Outputs:
            *r*: :class:`float`
                Half-width of coverage range
        :Versins:
            * 2018-09-28 ``@ddalle``: Version 1.0
            * 2020-02-21 ``@ddalle``: Rewritten from :mod:`cape.attdb.fm`
        """
        # Process search kwargs
        kw_find = {
            "mask": mask,
            "maskt": kw.pop("maskt", None),
            "once": True,
            "cols": kw.pop("searchcols", None),
            "tol": kw.get("tol", 1e-8),
            "tols": kw.get("tols", {}),
        }
        # Find indices of matches
        I, J = self.match(dbt, **kw_find)
        # Check for empty
        if I.size == 0:
            raise ValueError("No matches between databases")
        # Get values from this database
        V1 = self.get_values(col, I)
        # Get values from target database
        if isinstance(dbt, DataKit):
            # Get values with converters
            V2 = dbt.get_values(col, J)
        else:
            # Get values from dict
            V2 = dbt[col][J]
        # Deltas (unsigned)
        R = np.abs(V2 - V1)
        # Calculate interval
        return statutils.get_range(R, cov, **kw)

   # --- Integration ---
    # Integrate a 2D field
    def make_integral(self, col, xcol=None, ocol=None, **kw):
        r"""Integrate the columns of a 2D data col

        This method will not perform integration if *ocol* is already
        present in the database.

        :Call:
            >>> y = db.make_integral(col, xcol=None, ocol=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with analysis tools
            *col*: :class:`str`
                Name of data column to integrate
            *xcol*: {``None``} | :class:`str`
                Name of column to use as *x*-coords for integration
            *ocol*: {``col[1:]``} | :class:`str`
                Name of col to store result in
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Mask or indices of which cases to integrate
            *x*: {``None``} | :class:`np.ndarray`
                Optional 1D or 2D *x*-coordinates directly specified
            *dx*: {``1.0``} | :class:`float`
                Uniform spacing to use if *xcol* and *x* are not used
            *method*: |intmethods|
                Integration method or callable function taking two args
                like :func:`np.trapz`
        :Outputs:
            *y*: :class:`np.ndarray`
                1D array of integral of each column of *db[col]*
        :Versions:
            * 2020-06-10 ``@ddalle``: Version 1.0

        .. |intmethods| replace::
            {``"trapz"``} | ``"left"`` | ``"right"`` | **callable**
        """
        # Default column name
        if ocol is None:
            # Try to remove a "d" from *col* ("dCN" -> "CN")
            ocol = self.lstrip_colname(col, "d")
        # Check for name
        if ocol in self:
            # Return it
            return self[ocol]
        # Otherwise use *create*
        return self.create_integral(col, xcol, ocol, **kw)

    # Integrate a 2D field
    def create_integral(self, col, xcol=None, ocol=None, **kw):
        r"""Integrate the columns of a 2D data col

        :Call:
            >>> y = db.create_integral(col, xcol=None, ocol=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with analysis tools
            *col*: :class:`str`
                Name of data column to integrate
            *xcol*: {``None``} | :class:`str`
                Name of column to use as *x*-coords for integration
            *ocol*: {``col[1:]``} | :class:`str`
                Name of col to store result in
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Mask or indices of which cases to integrate
            *x*: {``None``} | :class:`np.ndarray`
                Optional 1D or 2D *x*-coordinates directly specified
            *dx*: {``1.0``} | :class:`float`
                Uniform spacing to use if *xcol* and *x* are not used
            *method*: |intmethods|
                Integration method or callable function taking two args
                like :func:`np.trapz`
        :Outputs:
            *y*: :class:`np.ndarray`
                1D array of integral of each column of *db[col]*
        :Versions:
            * 2020-03-24 ``@ddalle``: Version 1.0
            * 2020-06-02 ``@ddalle``: Added *mask*, callable *method*

        """
        # Default column name
        if ocol is None:
            # Try to remove a "d" from *col* ("dCN" -> "CN")
            ocol = self.lstrip_colname(col, "d")
        # Perform the integration
        y = self.genr8_integral(col, xcol, **kw)
        # Save column and definition
        self.save_col(ocol, y)
        # Output
        return y

    # Integrate a 2D field
    def genr8_integral(self, col, xcol=None, **kw):
        r"""Integrate the columns of a 2D data col

        :Call:
            >>> y = db.genr8_integral(col, xcol=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with analysis tools
            *col*: :class:`str`
                Name of data column to integrate
            *xcol*: {``None``} | :class:`str`
                Name of column to use as *x*-coords for integration
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Mask or indices of which cases to integrate
            *x*: {``None``} | :class:`np.ndarray`
                Optional 1D or 2D *x*-coordinates directly specified
            *dx*: {``1.0``} | :class:`float`
                Uniform spacing to use if *xcol* and *x* are not used
            *method*: |intmethods|
                Integration method or callable function taking two args
                like :func:`np.trapz`
        :Outputs:
            *y*: :class:`np.ndarray`
                1D array of integral of each column of *db[col]*
        :Versions:
            * 2020-03-24 ``@ddalle``: Version 1.0
            * 2020-06-02 ``@ddalle``: Added *mask*, callable *method*
            * 2020-06-04 ``@ddalle``: Split :func:`_genr8_integral`
        """
        # Get mask
        mask = kw.get("mask")
        # Get dimension of *col*
        ndim = self.get_col_prop(col, "Dimension")
        # Ensure 2D column
        if ndim != 2:
            raise ValueError("Col '%s' is not 2D" % col)
        # Get values
        v = self.get_values(col, mask)
        # Number of conditions
        nx = v.shape[0]
        # Process *x*
        if xcol is None:
            # Use 0, 1, 2, ... as *x* coords
            x = kw.get("x")
        else:
            # Get values
            x = self.get_all_values(xcol)
        # Get dimension
        if x is None:
            # Get "dx"
            dx = kw.get("dx")
            # Default
            if dx is None:
                dx = 1.0
            # Create default
            x = dx * np.arange(nx)
        # Calculate integral
        return self._genr8_integral(v, x, **kw)

    # Integrate a 2D field
    def _genr8_integral(self, v, x, method="trapz", **kw):
        r"""Integrate the columns of a 2D data col

        :Call:
            >>> y = db._genr8_integral(v, x, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with analysis tools
            *v*: :class:`np.ndarray`
                2D array of values to be integrated
            *x*: :class:`np.ndarray`
                1D or 2D *x*-coordinates directly specified
            *method*: |intmethods|
                Integration method or callable function taking two args
                like :func:`np.trapz`
        :Outputs:
            *y*: :class:`np.ndarray`
                1D array of integral of each column of *db[col]*
        :Versions:
            * 2020-06-04 ``@ddalle``: Forked from :func:`genr8_integral`

        .. |intmethods| replace::
            {``"trapz"``} | ``"left"`` | ``"right"`` | **callable**
        """
        # Check it
        if not (callable(method) or typeutils.isstr(method)):
            # Method must be string
            raise TypeError(
                "'method' must be 'str' or callable, got '%s'" % type(method))
        if method not in {"trapz", "left", "right"}:
            # Invalid name
            raise ValueError(
                ("method '%s' not supported; " % method) +
                ("options are 'trapz', 'left', 'right'"))
        # Number of conditions
        nx, ny = v.shape
        # Ensure array
        if not isinstance(x, np.ndarray):
            raise TypeError("x-coords for integration must be array")
        elif not isinstance(v, np.ndarray):
            raise TypeError("1D loads values must be an array")
        # Determine from *x*
        ndx = x.ndim
        # Ensure 1 or 2
        if ndx == 1:
            # Calculate *dx* vector beforehand
            dx = np.diff(x)
        elif ndx == 2:
            # Nothing to do here
            pass
        else:
            raise ValueError(
                "Integration does not support %i-D x coords" % ndx)
        # Initialize output
        y = np.zeros(ny, dtype=v.dtype)
        # Loop through conditions
        for i in range(ny):
            # Check method
            if method == "trapz":
                # Trapezoidal integration
                if ndx == 1:
                    # Common *x* coords
                    y[i] = np.trapz(v[:,i], x)
                else:
                    # Select *x* column
                    y[i] = np.trapz(v[:,i], x[:,i])
                # Go to next interval
                continue
            # Check *x* dimension
            if ndx == 2:
                # Select column and get intervale widhtds
                dx = np.diff(x[:,i])
            # Check L/R
            if method == "left":
                # Lower rectangular sum
                y[i] = np.sum(dx * v[:,i][:-1])
            elif method == "right":
                # Upper rectangular sum
                y[i] = np.sum(dx * v[:,i][1:])
            elif ndx == 1:
                # Callable with common *x*
                y[i] = method(v[:,i], x)
            else:
                # Callable with custom *x*
                y[i] = method(v[:,i], x[:,i])
        # Output
        return y
  # >

  # ===================
  # Plot
  # ===================
  # <
   # --- Preprocessors ---
    # Process arguments to plot_scalar()
    def _prep_args_plot1(self, *a, **kw):
        r"""Process arguments to :func:`plot_scalar`

        :Call:
            >>> col, I, J, a, kw = db._prep_args_plot1(*a, **kw)
            >>> col, I, J, a, kw = db._prep_args_plot1(I, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *a*: :class:`tuple`\ [:class:`np.ndarray` | :class:`float`]
                Array of values for arguments to :func:`db.__call__`
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Indices of exact entries to plot
            *kw*: :class:`dict`
                Keyword arguments to plot function and evaluation
        :Outputs:
            *col*: :class:`str`
                Data field to evaluate
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Indices of exact entries to plot
            *J*: :class:`np.ndarray`\ [:class:`int`]
                Indices of matches within *a*
            *a*: :class:`tuple`\ [:class:`float` | :class:`np.ndarray`]
                Values for arguments for *col* evaluator
            *kw*: :class:`dict`
                Processed keyword arguments with defaults applied
        :Versions:
            * 2019-03-14 ``@ddalle``: Version 1.0
            * 2019-12-26 ``@ddalle``: From :mod:`tnakit.db.db1`
            * 2020-03-27 ``@ddalle``: From :func:`_process_plot_args1`
        """
       # --- Argument Types ---
        # Process coefficient name and remaining coeffs
        col, a, kw = self._prep_args_colname(*a, **kw)
        # Get list of arguments
        arg_list = self.get_response_args(col)
        # Default arg
        if arg_list is None:
            # No default arg
            xk = None
            # Get key for *x* axis (don't set to ``None``)
            xk = kw.get("xk")
        else:
            # Default to first arg
            xk = arg_list[0]
            # Get key for *x* axis
            xk = kw.setdefault("xk", xk)
        # Set label
        if xk is None:
            # Default label becomes "Index"
            xlbl = "Index"
        else:
            # Default *x*-axis label is name of *x* key
            xlbl = xk
        # Check for indices
        if len(a) == 0:
            # Plot all values
            mask = None
        else:
            # Process first second arg as a mask
            mask = a[0]
        # Check if it looks like a mask
        qmask = self.check_mask(mask)
        # Check for integer
        if qmask:
            # Turn it into indices
            I = self.prep_mask(mask, col=col)
            # Request for exact values
            qexact  = True
            qinterp = False
            qmark   = False
            qindex  = True
            # Get values of arg list from *DBc* and *I*
            if arg_list:
                # Initialize
                A = []
                # Loop through *response_args*
                for arg in arg_list:
                    # Get values
                    A.append(self.get_xvals(arg, I, **kw))
                # Convert to tuple
                a = tuple(A)
            elif xk:
                # Use the *xk* only
                a = (self.get_xvals(xk, I, **kw),)
            else:
                # For trivial case, eval arg is just *I*
                a = (I,)
            # Plot all points
            J = np.arange(I.size)
        else:
            # No request for exact values
            qexact  = False
            qinterp = True
            qmark   = True
            qindex  = False
            # Find matches from *a to database points
            I, J = self.find(arg_list, *a, **kw)
       # --- Options: What to plot ---
        # Plot exact values, interpolated (eval), and markers of actual data
        qexact  = kw.setdefault("PlotExact",  qexact)
        qinterp = kw.setdefault("PlotInterp", qinterp and (not qexact))
        qmark   = kw.setdefault("MarkExact",  qmark and (not qexact))
        # Default UQ coefficient
        uk_def = self.get_uq_col(col)
        # Check situation
        if typeutils.isarray(uk_def):
            # Get first entry
            uk_def = uk_def[0]
        # Get UQ coefficient
        uk  = kw.get("uk",  kw.get("ucol"))
        ukM = kw.get("ukM", kw.get("ucol_minus", uk))
        ukP = kw.get("ukP", kw.get("ucol_plus",  uk))
        # Turn on *PlotUQ* if UQ key specified
        if ukM or ukP:
            kw.setdefault("ShowUncertainty", True)
        # UQ flag
        quq = kw.get("ShowUncertainty", kw.get("ShowUQ", False))
        # Set default UQ keys if needed
        if quq:
            uk  = kw.setdefault("uk",  uk_def)
            ukM = kw.setdefault("ukM", uk)
            ukP = kw.setdefault("ukP", uk)
       # --- Default Labels ---
        # Default label starter: db.name
        dlbl = self.__dict__.get("name")
        # Some fallbacks
        if dlbl is None:
            dlbl = self.__dict__.get("comp")
        if dlbl is None:
            dlbl = self.__dict__.get("Name")
        if dlbl is None:
            dlbl = col
        # Set default label
        kw.setdefault("Label", dlbl)
        # Default x-axis label is *xk*
        kw.setdefault("XLabel", xlbl)
        kw.setdefault("YLabel", col)
       # --- Cleanup ---
        # Output
        return col, I, J, a, kw

    # Process arguments to plot_linear()
    def _prep_args_plot2(self, *a, **kw):
        r"""Process arguments to :func:`plot_linear`

        :Call:
            >>> col, X, V, kw = db._prep_args_plot2(*a, **kw)
            >>> col, X, V, kw = db._prep_args_plot2(I, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *a*: :class:`tuple`\ [:class:`np.ndarray` | :class:`float`]
                Array of values for arguments to :func:`db.__call__`
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Indices of exact entries to plot
            *kw*: :class:`dict`
                Keyword arguments to plot function and evaluation
        :Outputs:
            *col*: :class:`str`
                Data field to evaluate
            *X*: :class:`np.ndarray`\ [:class:`float`]
                *x*-values for 1D output sensitivity
            *V*: :class:`np.ndarray`\ [:class:`float`]
                Values of *col* for cases *I*
            *kw*: :class:`dict`
                Processed keyword arguments with defaults applied
        :Versions:
            * 2020-03-27 ``@ddalle``: Version 1.0
        """
       # --- Argument Types ---
        # Process coefficient name and remaining coeffs
        col, a, kw = self._prep_args_colname(*a, **kw)
        # Get list of arguments
        args = self.get_response_args(col)
        # Get *xk* for output
        xarg = self.get_output_xarg1(col)
        # Get key for *x* axis
        xk = kw.setdefault("xk", xarg)
        xk = kw.get("xcol", xk)
        # Check for indices
        if len(a) == 0:
            raise ValueError("At least 2 inputs required; received 1")
        # Get dimension of xarg
        ndimx = self.get_ndim(xk)
        # Get first entry to determine index vs values
        I = np.asarray(a[0])
        # Data types for first input and first response_arg
        dtypeI = I.dtype.name
        dtype0 = self.get_col_dtype(col)
        # Check for integer
        qintI = dtypeI.startswith("int") or dtypeI.startswith("uint")
        # Check for first arg integer
        qint0 = dtype0.startswith("int") or dtype0.startswith("uint")
        # If first *arg* is int, try to parse all values
        if qint0:
            # If it fails, use indices
            try:
                # Loop through args
                for j, arg in enumerate(args):
                    # Get arg value
                    aj = self.get_arg_value(j, arg, *a, **kw)
            except Exception:
                # Didn't work; assume indices
                qindex = True
            else:
                # Worked; assume values
                qindex = False
        else:
            # First *response_arg* not int; check *a[0]* for int
            qindex = qintI
        # Check data
        if qindex:
            # Get values
            V = self.get_all_values(col)[:,I]
            # Get *x* values
            if ndimx == 2:
                # Get *x* values for each index
                X = self.get_all_values(xk)[:,I]
            else:
                # Common *x* values
                X = self.get_all_values(xk)
        else:
            # Evaluate
            V = self(col, *a, **kw)
            # Get *x* values
            if ndimx == 2:
                # Get *x* values for each index
                X = self(xk, *a, **kw)
            else:
                # Common *x* values
                X = self.get_all_values(xk)
       # --- Options ---
        # Default label starter: db.name
        dlbl = self.__dict__.get("name")
        # Some fallbacks
        if dlbl is None:
            dlbl = self.__dict__.get("comp")
        if dlbl is None:
            dlbl = self.__dict__.get("Name")
        if dlbl is None:
            dlbl = col
        # Set default label
        kw.setdefault("Label", dlbl)
        # Default x-axis label is *xk*
        kw.setdefault("XLabel", xk)
        kw.setdefault("YLabel", col)
       # --- Cleanup ---
        # Output
        return col, X, V, kw

   # --- Base Plot Commands ---
    # Master plot controller
    def plot(self, *a, **kw):
        r"""Plot a scalar or linear data column

        This function tests the output dimension of *col*.  For a
        standard data column, which is a scalar, this will pass the
        args to :func:`plot_scalar`.  If ``db.get_ndim(col)`` is ``2``,
        however (for example a line load), :func:`plot_linear` will be
        called instead.

        :Call:
            >>> h = db.plot(col, *a, **kw)
            >>> h = db.plot(col, I, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Data column (or derived column) to evaluate
            *a*: :class:`tuple`\ [:class:`np.ndarray` | :class:`float`]
                Array of values for arguments to evaluator for *col*
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Indices of exact entries to plot
            *xcol*, *xk*: :class:`str`
                Key/column name for *x* axis
        :Keyword Arguments:
            %(keys)s
        :Outputs:
            *h*: :class:`plot_mpl.MPLHandle`
                Object of :mod:`matplotlib` handles
        :See Also:
            * :func:`DataKit.plot_scalar`
            * :func:`DataKit.plot_linear`
            * :func:`cape.tnakit.plot_mpl.plot`
        :Versions:
            * 2020-04-20 ``@ddalle``: Version 1.0
        """
        # Process column name and remaining coeffs
        col, a, kw = self._prep_args_colname(*a, **kw)
        # Get dimension of *col*
        ndim = self.get_ndim(col)
        # Check dimension
        if ndim is None:
            # Column not found
            raise KeyError("No col '%s' found" % col)
        elif ndim == 1:
            # Scalar plot
            return self.plot_scalar(col, *a, **kw)
        elif ndim == 2:
            # Line load plot
            return self.plot_linear(col, *a, **kw)
        else:
            # Not implemented
            raise ValueError("No plot method for %iD col '%s'" % (ndim, col))

    # Plot a sweep of one or more coefficients
    def plot_scalar(self, *a, **kw):
        r"""Plot a sweep of one data column over several cases

        This is the base method for plotting scalar *col*\ s. Other
        methods may call this one with modifications to the default
        settings.

        :Call:
            >>> h = db.plot_scalar(col, *a, **kw)
            >>> h = db.plot_scalar(col, I, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Data column (or derived column) to evaluate
            *a*: :class:`tuple`\ [:class:`np.ndarray` | :class:`float`]
                Array of values for arguments to evaluator for *col*
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Indices of exact entries to plot
            *xcol*, *xk*: {``None``} | :class:`str`
                Key/column name for *x* axis
            *PlotExact*: ``True`` | ``False``
                Plot exact values directly from database without
                interpolation. Default is ``True`` if *I* is used
            *PlotInterp*: ``True`` | ``False``
                Plot values by using :func:`DBc.__call__`
            *MarkExact*: ``True`` | ``False``
                Mark interpolated curves with markers where actual data
                points are present
        :Keyword Arguments:
            %(keys)s
        :Outputs:
            *h*: :class:`plot_mpl.MPLHandle`
                Object of :mod:`matplotlib` handles
        :Versions:
            * 2015-05-30 ``@ddalle``: Version 1.0
            * 2015-12-14 ``@ddalle``: Added error bars
            * 2019-12-26 ``@ddalle``: From :mod:`tnakit.db.db1`
            * 2020-03-30 ``@ddalle``: Redocumented
        """
       # --- Process Args ---
        # Process coefficient name and remaining coeffs
        col, I, J, a, kw = self._prep_args_plot1(*a, **kw)
        # Get list of arguments
        arg_list = self.get_response_args(col)
        # Get key for *x* axis
        if arg_list:
            # Default to first *arg*
            xk = arg_list[0]
        else:
            # No key
            xk = None
        # Check for direct specification
        xk = kw.pop("xcol", kw.pop("xk", xk))
       # --- Options: What to plot ---
        # Plot exact values, interpolated (eval), and markers of actual data
        qexact  = kw.pop("PlotExact",  False)
        qinterp = kw.pop("PlotInterp", True)
        qmark   = kw.pop("MarkExact",  True)
        # Default UQ coefficient
        uk_def = self.get_uq_col(col)
        # Ensure string
        if typeutils.isarray(uk_def):
            uk_def = uk_def[0]
        # Get UQ coefficient
        uk  = kw.pop("ucol",       kw.pop("uk", uk_def))
        ukM = kw.pop("ucol_minus", kw.pop("ukM", uk))
        ukP = kw.pop("ucol_plus",  kw.pop("ukP", uk))
       # --- Plot Values ---
        # Initialize output
        h = pmpl.MPLHandle()
        # Initialize plot options in order to reduce aliases, etc.
        opts = pmpl.MPLOpts(_warnmode=0, **kw)
        # Uncertainty plot flag
        quq = opts.get("ShowUncertainty", False)
        # Y-axis values: exact
        if qexact:
            # Get corresponding *x* values
            if xk is None:
                # Just use index
                xe = I
            else:
                # Get values
                xe = self.get_xvals(xk, I, **kw)
            # Try to get values directly from database
            ye = self.get_yvals_exact(col, I, **kw)
            # Evaluate UQ-minus
            if quq and ukM:
                # Get UQ value below
                uyeM = self.rcall_from_index(ukM, I, **kw)
            elif quq:
                # Use zeros for negative error term
                uyeM = np.zeros_like(ye)
            # Evaluate UQ-pluts
            if quq and ukP and ukP==ukM:
                # Copy negative terms to positive
                uyeP = uyeM
            elif quq and ukP:
                # Evaluate separate UQ above
                uyeP = self.rcall_from_index(ukP, I, **kw)
            elif quq:
                # Use zeros
                uyeP = np.zeros_like(ye)
        # Y-axis values: evaluated/interpolated
        if qmark or qinterp:
            # Get values for *x*-axis
            xv = self.get_xvals_eval(xk, col, *a, **kw)
            # Evaluate function
            yv = self.__call__(col, *a, **kw)
            # Evaluate UQ-minus
            if quq and ukM:
                # Get UQ value below
                uyM = self.rcall_from_arglist(ukM, arg_list, *a, **kw)
            elif quq:
                # Use zeros for negative error term
                uyM = np.zeros_like(yv)
            # Evaluate UQ-pluts
            if quq and ukP and ukP==ukM:
                # Copy negative terms to positive
                uyP = uyM
            elif quq and ukP:
                # Evaluate separate UQ above
                uyP = self.rcall_from_arglist(ukP, arg_list, *a, **kw)
            elif quq:
                # Use zeros
                uyP = np.zeros_like(yv)
       # --- Data Cleanup ---
        # Create input to *markevery*
        if qmark:
            # Check length
            if J.size == xv.size:
                # Mark all cases
                marke = None
            else:
                # Convert to list
                marke = list(J)
        # Remove extra keywords if possible
        if arg_list:
            for k in arg_list:
                kw.pop(k, None)
       # --- Primary Plot ---
        # Initialize output
        h = pmpl.MPLHandle()
        # Initialize plot options in order to reduce aliases, etc.
        opts = pmpl.MPLOpts(**kw)
        # Initialize plot options and get them locally
        kw_p = opts.setdefault("PlotOptions", {})
        # Create a copy
        kw_p0 = dict(kw_p)
        # Existing uncertainty plot type
        t_uq = opts.get("UncertaintyPlotType")
        # Marked and interpolated data
        if qinterp or qmark:
            # Default marker setting
            if qmark:
                kw_p.setdefault("marker", "^")
            else:
                kw_p.setdefault("marker", "")
            # Default line style
            if not qinterp:
                # Turn off lines
                kw_p.setdefault("ls", "")
                # Default UQ style is error bars
                if t_uq is None:
                    opts["UncertaintyPlotType"] = "errorbar"
            # Check for uncertainty
            if quq:
                # Set values
                opts["uy"] = np.array([uyM, uyP])
            # Call the main function
            hi = pmpl.plot(xv, yv, **opts)
            # Apply markers
            if qmark:
                # Get line handle
                hl = hi.lines[0]
                # Apply which indices to mark
                hl.set_markevery(marke)
            # Combine
            h.add(hi)
       # --- Exact Plot ---
        # Plot exact values
        if qexact:
            # Turn on marker
            if "marker" not in kw_p0:
                kw_p["marker"] = "^"
            # Turn *lw* off
            if "ls" not in kw_p0:
                kw_p["ls"] = ""
            # Set UQ style to "errorbar"
            if t_uq is None:
                opts["UncertaintyPlotType"] = "errorbar"
            # Check for uncertainty
            if quq:
                # Set values
                opts["yerr"] = np.array([uyeM, uyeP])
            # Plot exact data
            he = pmpl.plot(xe, ye, **opts)
            # Combine
            h.add(he)
       # --- Cleanup ---
        # Output
        return h
       # ---

    # Plot raw data
    def plot_raw(self, x, y, **kw):
        r"""Plot 1D data sets directly, without response functions

        :Call:
            >>> h = db.plot_raw(xk, yk, **kw)
            >>> h = db.plot_raw(xv, yv, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *xk*: :class:`str`
                Name of *col* to use for x-axis
            *yk*: :class:`str`
                Name of *col* to use for y-axis
            *xv*: :class:`np.ndarray`
                Directly specified values for x-axis
            *yv*: :class:`np.ndarray`
                Directly specified values for y-axis
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Mask of which points to include in plot
        :Outputs:
            *h*: :class:`plot_mpl.MPLHandle`
                Object of :mod:`matplotlib` handles
        :Versions:
            * 2020-12-31 ``@ddalle``: Version 1.0
        """
        # Get *mask* for plotting subset
        mask = kw.pop("mask", None)
        # Initialize plot options in order to reduce aliases, etc.
        opts = pmpl.MPLOpts(_warnmode=0, **kw)
        # Check for existing handle
        h = kw.get("h")
        # Initialize output
        if h is None:
            # Create an empty one
            h = pmpl.MPLHandle()
        # Get *x* values to plot
        if typeutils.isstr(x):
            # Get values
            xv = self.get_values(x, mask)
            # Check validity
            if xv is None:
                raise KeyError("Could not find col '%s'" % x)
            # Save default labels
            opts.setdefault_option("XLabel", x)
        else:
            # Assume it already is an array of values
            xv = x
        # Get *y* values to plot
        if typeutils.isstr(y):
            # Get values
            yv = self.get_values(y, mask)
            # Check validity
            if yv is None:
                raise KeyError("Could not find col '%s'" % y)
            # Save default labels
            opts.setdefault_option("YLabel", y)
        else:
            # Assume it already is an array of values
            yv = y
        # Just plot it with minimal changes
        hi = pmpl.plot(xv, yv, **opts)
        # Append to handle
        h.add(hi)
        # Output
        return h

    # Plot single line load
    def plot_linear(self, *a, **kw):
        r"""Plot a 1D-output col for one or more cases or conditions

        :Call:
            >>> h = db.plot_linear(col, *a, **kw)
            >>> h = db.plot_linear(col, I, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Data column (or derived column) to evaluate
            *a*: :class:`tuple`\ [:class:`np.ndarray` | :class:`float`]
                Array of values for arguments to evaluator for *col*
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Indices of exact entries to plot
            *xcol*, *xk*: {*db.response_xargs[col][0]*} | :class:`str`
                Key/column name for *x* axis
        :Keyword Arguments:
            %(keys)s
        :Outputs:
            *h*: :class:`plot_mpl.MPLHandle`
                Object of :mod:`matplotlib` handles
        :Versions:
            * 2020-03-30 ``@ddalle``: Version 1.0
        """
       # --- Prep ---
        # Process column name and values to plot
        col, X, V, kw = self._prep_args_plot2(*a, **kw)
        # Get key for *x* axis
        xk = kw.pop("xcol", kw.pop("xk", self.get_output_xarg1(col)))
       # --- Plot Values ---
        # Check for existing handle
        h = kw.get("h")
        # Initialize output
        if h is None:
            # Create an empty one
            h = pmpl.MPLHandle()
        # Initialize plot options in order to reduce aliases, etc.
        opts = pmpl.MPLOpts(_warnmode=0, **kw)
        # Check dimension of *V*
        if V.ndim == 1:
            # Scalar line plot
            hi = pmpl.plot(X, V, **opts)
            # Combine plot handles
            h.add(hi)
        elif X.ndim == 1:
            # Multiple conditions with common *x*
            for i in range(V.shape[1]):
                # Line plot for column of *V*
                hi = pmpl.plot(X, V[:,i], Index=i, **opts)
                # Combine plot handles
                h.add(hi)
        else:
            # Multiple conditions with common *x*
            for i in range(V.shape[1]):
                # Line plot for column of *V*
                hi = pmpl.plot(X[:, i], V[:,i], Index=i, **opts)
                # combine plot handles
                h.add(hi)
       # --- PNG ---
        # Plot the png image if appropriate
        if kw.get("ShowPNG", True):
            h = self.plot_png(col, fig=h.fig, h=h)
       # --- Seam Curve ---
        # Plot the seam curve if appropriate
        if kw.get("ShowSeam", True):
            h = self.plot_seam(col, fig=h.fig, h=h)
       # --- Output ---
        # Return plot handle
        return h
       # ---

   # --- Semilogy ---
    # Plot raw data on log-y scale
    def semilogy_raw(self, x, y, **kw):
        r"""Plot 1D data sets directly, without response functions

        :Call:
            >>> h = db.semilogy_raw(xk, yk, **kw)
            >>> h = db.semilogy_raw(xv, yv, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *xk*: :class:`str`
                Name of *col* to use for x-axis
            *yk*: :class:`str`
                Name of *col* to use for y-axis
            *xv*: :class:`np.ndarray`
                Directly specified values for x-axis
            *yv*: :class:`np.ndarray`
                Directly specified values for y-axis
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Mask of which points to include in plot
        :Outputs:
            *h*: :class:`plot_mpl.MPLHandle`
                Object of :mod:`matplotlib` handles
        :Versions:
            * 2021-01-05 ``@ddalle``: Version 1.0; fork :func:`plot_raw`
        """
        # Get *mask* for plotting subset
        mask = kw.pop("mask", None)
        # Initialize plot options in order to reduce aliases, etc.
        opts = pmpl.MPLOpts(**kw)
        # Get *x* values to plot
        if typeutils.isstr(x):
            # Get values
            xv = self.get_values(x, mask)
            # Save default labels
            opts.setdefault_option("XLabel", x)
        else:
            # Assume it already is an array of values
            xv = x
        # Get *y* values to plot
        if typeutils.isstr(y):
            # Get values
            yv = self.get_values(y, mask)
            # Save default labels
            opts.setdefault_option("YLabel", y)
        else:
            # Assume it already is an array of values
            yv = y
        # Just plot it
        return pmpl.semilogy(xv, yv, **opts)

   # --- Contour ---
    # Plot contour
    def plot_contour(self, *a, **kw):
        r"""Create a contour plot of one *col* vs two others

        :Call:
            >>> h = db.plot_contour(col, *a, **kw)
            >>> h = db.plot_contour(col, mask, **kw)
            >>> h = db.plot_contour(col, mask_index, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Data column (or derived column) to evaluate
            *a*: :class:`tuple`\ [:class:`np.ndarray` | :class:`float`]
                Array of values for arguments to evaluator for *col*
            *mask*: :class:`np.ndarray`\ [:class:`bool`]
                Mask of which points to include in plot
            *mask_index*: :class:`np.ndarray`\ [:class:`int`]
                Indices of points to include in plot
            *xcol*, *xk*: :class:`str`
                Name of column to use for *x* axis
            *ycol*, *yk*: :class:`str`
                Name of column to use for *y* axis
        :Keyword Arguments:
            %(keys)s
            %(axkeys)s
        :Outputs:
            *h*: :class:`plot_mpl.MPLHandle`
                Object of :mod:`matplotlib` handles
        :Versions:
            * 2020-04-24 ``@ddalle``: Version 1.0
        """
       # --- Argument Types ---
        # Process coefficient name and remaining coeffs
        col, a, kw = self._prep_args_colname(*a, **kw)
        # Separate out rcall() and plot() kwargs
        kwr, kwo = self.sep_response_kwargs(col, **kw)
       # --- Value Lookup ---
        # Check call mode
        mode = self._check_callmode(col, *a, **kw)
        # Response args
        args = self.get_response_args(col)
        # Number of args
        if args:
            # Nontrivial response args set
            narg = len(args)
        else:
            # No args
            narg = 0
        # Default cols for axes
        if narg > 0:
            xk = args[0]
        else:
            xk = None
        if narg > 1:
            yk = args[1]
        else:
            yk = None
        # Get cols to use for axes
        xk = kwo.pop("xcol", kwo.pop("xk", xk))
        yk = kwo.pop("ycol", kwo.pop("yk", xk))
        # These are required
        if xk is None:
            raise ValueError("No required 'xcol' option specified")
        if yk is None:
            raise ValueError("No required 'ycol' option specified")
        # Get main values
        v = self(col, *a, **kwr)
        # Check for errors
        if v is None:
            raise ValueError("Unable to evaluate col '%s'" % col)
        # Lookup main args
        if mode == 0:
            # All values
            x = self.get_all_values(xk)
            y = self.get_all_values(yk)
        elif mode == 1:
            # Indices/mask
            x = self.get_values(xk, a[0])
            y = self.get_values(yk, a[0])
        elif mode in [2, 3]:
            # Check if this is an arg
            if xk not in args:
                raise ValueError(
                    "Cannot determine values of '%s' from args to '%s'"
                    % (xk, col))
            if yk not in args:
                raise ValueError(
                    "Cannot determine values of '%s' from args to '%s'"
                    % (xk, col))
            # Evaluation args
            x = self.get_arg_value(args.index(xk), xk, *a, **kwr)
            y = self.get_arg_value(args.index(yk), yk, *a, **kwr)
        else:
            # Couldn't figure out *x* and *y*
            raise ValueError("Could not determine lookup mode for '%s'" % col)
       # --- Plot ---
        # Default labels
        kwo.setdefault("XLabel", xk)
        kwo.setdefault("YLabel", yk)
        # Set default for *MarkPoints*
        if mode < 2:
            # Plotting exact points
            kwo.setdefault("MarkPoints", True)
        else:
            # Plotting output from response
            kwo.setdefault("MarkPoints", False)
        # Call contour function
        return pmpl.contour(x, y, v, **kwo)

   # --- PNG ---
    # Plot PNG
    def plot_png(self, col, fig=None, h=None, **kw):
        r"""Show tagged PNG image in new axes

        :Call:
            >>> h = db.plot_png(col, fig=None, h=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column being plotted
            *png*: {*db.cols_png[col]*} | :class:`str`
                Name used to tag this PNG image
            *fig*: {``None``} | :class:`Figure` | :class:`int`
                Name or number of figure in which to plot image
            *h*: {``None``} | :class:`cape.tnakit.plot_mpl.MPLHandle`
                Optional existing handle to various plot objects
        :Outputs:
            *h*: :class:`cape.tnakit.plot_mpl.MPLHandle`
                Plot object container
            *h.img*: :class:`matplotlib.image.AxesImage`
                PNG image object
            *h.ax_img*: :class:`AxesSubplot`
                Axes handle in wich *h.img* is shown
        :Versions:
            * 2020-04-02 ``@ddalle``: Version 1.0
        """
        # Get name of PNG to add
        png = self.get_col_png(col)
        # Check for override from *kw*
        png = kw.pop("png", png)
        # Default handle
        if h is None:
            # Create one
            h = pmpl.MPLHandle()
            # Create axes
            h.ax = pmpl.get_axes(fig)
        # Exit if None
        if png is None:
            # Nothing to show
            return h
        # Get figure handle (from ``None``, handle, or number)
        fig = pmpl.get_figure(fig)
        # Check if already plotted
        if self.check_png_fig(png, fig):
            # Already plotted
            return h
        # Get axes
        ax_png = fig.add_subplot(212)
        # Get name of image file to show
        fpng = self.get_png_fname(png)
        # Get plot kwargs
        kw_png = self.get_png_kwargs(png)
        # Show the image nicely
        img = pmpl.imshow(fpng, **kw_png)
        # Steal the x-axis label
        xlbl = h.ax.get_xlabel()
        # Shift it from main plot to PNG plot
        ax_png.set_xlabel(xlbl)
        h.ax.set_xlabel("")
        # Turn off x-coord labels too
        h.ax.set_xticklabels([])
        # Format extents nicely
        pmpl.axes_adjust_col(h.fig, SubplotRubber=h.ax)
        # Turn off aspect ratio
        ax_png.set_aspect("auto")
        # Get current limits
        xmina, xmaxa = h.ax.get_xlim()
        xminb, xmaxb = ax_png.get_xlim()
        yminb, ymaxb = ax_png.get_ylim()
        # Reset *ylims* for PNG image
        h2 = (ymaxb - yminb) / (xmaxb - xminb) * (xmaxa - xmina)
        ymin2 = 0.5*(yminb + ymaxb - h2)
        ymax2 = 0.5*(yminb + ymaxb + h2)
        # Tie horizontal limits
        ax_png.set_xlim(xmina, xmaxa)
        ax_png.set_ylim(ymin2, ymax2)
        # Turn aspect ratio back on?
        ax_png.set_aspect("equal")
        # Label the axes
        ax_png.set_label("<img>")
        # Save parameters
        h.fig = fig
        h.img = img
        h.ax_img = ax_png
        # Save this image in list for PNG tag
        self.add_png_fig(png, fig)
        # Reset current axes
        pmpl.mpl.plt.sca(h.ax)
        # Output
        return h

   # --- Seam ---
    # Plot seam curve
    def plot_seam(self, col, fig=None, h=None, **kw):
        r"""Show tagged seam curve in new axes

        :Call:
            >>> h = db.plot_seam(col, fig=None, h=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Name of data column being plotted
            *png*: {*db.cols_png[col]*} | :class:`str`
                Name used to tag this PNG image
            *fig*: {``None``} | :class:`Figure` | :class:`int`
                Name or number of figure in which to plot image
            *h*: {``None``} | :class:`cape.tnakit.plot_mpl.MPLHandle`
                Optional existing handle to various plot objects
        :Outputs:
            *h*: :class:`cape.tnakit.plot_mpl.MPLHandle`
                Plot object container
            *h.lines_seam*: :class:`list`\ [:class:`matplotlib.Line2D`]
                Seam curve handle
            *h.ax_seam*: :class:`AxesSubplot`
                Axes handle in wich *h.seam* is shown
        :Versions:
            * 2020-04-02 ``@ddalle``: Version 1.0
        """
        # Get name of seam curve to add
        seam = self.get_col_seam(col)
        # Check for override from *kw*
        seam = kw.pop("seam", seam)
        # Default handle
        if h is None:
            # Create one
            h = pmpl.MPLHandle()
            # Create axes
            h.ax = pmpl.get_axes(fig)
        # Exit if None
        if seam is None:
            # Nothing to show
            return h
        # Get figure handle (from ``None``, handle, or number)
        fig = pmpl.get_figure(fig)
        # Check if already plotted
        if self.check_seam_fig(seam, fig):
            # Already plotted
            return h
        # Get axes
        ax_seam = fig.add_subplot(212)
        # Get col names for seam
        xcol, ycol = self.get_seam_col(seam)
        # Get plot kwargs
        kw_seam = self.get_seam_kwargs(seam)
        # Get seam offset kwarg
        dy = kw_seam.pop("SeamDY", 0.0)
        # Plot the image
        hseam = pmpl.plot(self[xcol], self[ycol], **kw_seam)
        # Rescale
        pmpl.axes_autoscale_height(ax_seam)
        # Steal the x-axis label
        xlbl = h.ax.get_xlabel()
        # Shift it from main plot to seam plot
        ax_seam.set_xlabel(xlbl)
        h.ax.set_xlabel("")
        # Turn off x-coord labels too
        h.ax.set_xticklabels([])
        # Format extents nicely
        pmpl.axes_adjust_col(h.fig, SubplotRubber=h.ax)
        # Get current limits and output limits
        xmin0, xmax0 = ax_seam.get_xlim()
        ymin0, ymax0 = ax_seam.get_ylim()
        # Data *x* lims
        xmin, xmax = h.ax.get_xlim()
        # Scale *y* lims
        hy = (ymax0 - ymin0) * (xmax - xmin) / (xmax0 - xmin0)
        # Spread out
        ymin = dy + 0.5*(ymin0 + ymax0 - hy)
        ymax = dy + 0.5*(ymin0 + ymax0 + hy)
        # Tie horizontal limits
        ax_seam.set_xlim(xmin, xmax)
        # Update horizontal limits
        ax_seam.set_ylim(ymin, ymax)
        # Label the axes
        ax_seam.set_label("<seam>")
        # Save parameters
        h.fig = fig
        h.lines_seam = hseam.lines
        h.ax_seam = ax_seam
        # Save this image in list for seam tag
        self.add_seam_fig(seam, fig)
        # Reset current axes
        pmpl.mpl.plt.sca(h.ax)
        # Output
        return h

   # --- PNG Options: Get ---
    # Get PNG file name
    def get_png_fname(self, png):
        r"""Get name of PNG file

        :Call:
            >>> fpng = db.get_png_fname(png)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *png*: :class:`str`
                Name used to tag this PNG image
        :Outputs:
            *fpng*: ``None`` | :class:`str`
                Name of PNG file, if any
        :Versions:
            * 2020-04-02 ``@ddalle``: Version 1.0
        """
        # Check types
        if not typeutils.isstr(png):
            raise TypeError(
                "PNG name must be str (got %s)" % type(png))
        # Get handle to attribute
        png_fnames = self.__dict__.setdefault("png_fnames", {})
        # Get parameter
        return png_fnames.get(png)

    # Get plot kwargs for named PNG
    def get_png_kwargs(self, png):
        r"""Set evaluation keyword arguments for PNG file

        :Call:
            >>> kw = db.set_png_kwargs(png)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *png*: :class:`str`
                Name used to tag this PNG image
        :Outputs:
            *kw*: {``{}``} | :class:`MPLOpts`
                Options to use when showing PNG image (copied)
        :Versions:
            * 2020-04-02 ``@ddalle``: Version 1.0
        """
        # Get handle to kw
        png_kwargs = self.__dict__.setdefault("png_kwargs", {})
        # Check types
        if not typeutils.isstr(png):
            raise TypeError(
                "PNG name must be str (got %s)" % type(png))
        # Get kwargs
        kw = png_kwargs.get(png, {})
        # Create a copy
        return copy.copy(kw)

    # Get PNG tag to use for a given col
    def get_col_png(self, col):
        r"""Get name/tag of PNG image to use when plotting *col*

        :Call:
            >>> png = db.set_col_png(col)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Data column for to associate with *png*
        :Outputs:
            *png*: ``None`` | :class:`str`
                Name/abbreviation/tag of PNG image to use
        :Versions:
            * 2020-04-02 ``@ddalle``: Version 1.0
        """
        # Check types
        if not typeutils.isstr(col):
            raise TypeError(
                "Data column must be str (got %s)" % type(col))
        # Get handle to attribute
        col_pngs = self.__dict__.setdefault("col_pngs", {})
        # Get PNG name
        return col_pngs.get(col)

    # Check figure handle if it's in current list
    def check_png_fig(self, png, fig):
        r"""Check if figure is in set of active figs for PNG tag

        :Call:
            >>> q = db.check_png_fig(png, fig)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *png*: :class:`str`
                Name/abbreviation/tag of PNG image to use
            *fig*: ``None`` | :class:`matplotlib.figure.Figure`
                Figure handle
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not *fig* is in *db.png_figs[png]*
        :Versions:
            * 2020-04-01 ``@ddalle``: Version 1.0
        """
        # Return ``False`` if *fig* is ``None``
        if fig is None:
            return False
        # Get attribute
        png_figs = self.__dict__.get("png_figs", {})
        # Get current handles
        figs = png_figs.get(png, set())
        # Check for figure
        return fig in figs

   # --- PNG Options: Set ---
    # Build overall PNG description
    def make_png(self, png, fpng, cols=None, **kw):
        r"""Set all parameters to describe PNG image

        :Call:
            >>> db.make_png(png, fpng, cols, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *png*: :class:`str`
                Name used to tag this PNG image
            *fpng*: :class:`str`
                Name of PNG file
            *kw*: {``{}``} | :class:`dict`
                Options to use when showing PNG image
        :See Also:
            * :func:`set_cols_png`
            * :func:`set_png_fname`
            * :func:`set_png_kwargs`
        :Versions:
            * 2020-04-02 ``@ddalle``: Version 1.0
        """
        # Set file name
        self.set_png_fname(png, fpng)
        # Set plot *kwargs*
        self.set_png_kwargs(png, **kw)
        # Set *png* as tag for *cols*
        if cols:
            # Set for each column
            self.set_cols_png(cols, png)

    # Set PNG to use for list of *cols*
    def set_cols_png(self, cols, png):
        r"""Set name/tag of PNG image for several data columns

        :Call:
            >>> db.set_cols_png(cols, png)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *cols*: :class:`list`\ [:class:`str`]
                Data column for to associate with *png*
            *png*: :class:`str`
                Name/abbreviation/tag of PNG image to use
        :Effects:
            *db.col_pngs*: :class:`dict`
                Entry for *col* in *cols* set to *png*
        :Versions:
            * 2020-04-01 ``@ddalle``: Version 1.0
        """
        # Check input
        if not isinstance(cols, (tuple, list)):
            raise TypeError(
                "List of cols must be 'list' (got '%s')" % type(cols))
        # Check each col
        for (j, col) in enumerate(cols):
            if not typeutils.isstr(col):
                raise TypeError(
                    "col %i must be 'str' (got '%s')" % (j, type(col)))
        # Loop through columns
        for col in cols:
            # Call individual function
            self.set_col_png(col, png)

    # Set PNG tag to use for a given col
    def set_col_png(self, col, png):
        r"""Set name/tag of PNG image to use when plotting *col*

        :Call:
            >>> db.set_col_png(col, png)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Data column for to associate with *png*
            *png*: :class:`str`
                Name/abbreviation/tag of PNG image to use
        :Effects:
            *db.col_pngs*: :class:`dict`
                Entry for *col* set to *png*
        :Versions:
            * 2020-04-01 ``@jmeeroff``: Version 1.0
        """
        # Check types
        if not typeutils.isstr(png):
            raise TypeError(
                "PNG name must be str (got %s)" % type(png))
        if not typeutils.isstr(col):
            raise TypeError(
                "Data column must be str (got %s)" % type(col))
        # Get handle to attribute
        col_pngs = self.__dict__.setdefault("col_pngs", {})
        # Check type
        if not isinstance(col_pngs, dict):
            raise TypeError("col_pngs attribute is not a dict")
        # Set parameter (to a copy)
        col_pngs[col] = png

    # Set PNG file name
    def set_png_fname(self, png, fpng):
        r"""Set name of PNG file

        :Call:
            >>> db.set_png_fname(png, fpng)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *png*: :class:`str`
                Name used to tag this PNG image
            *fpng*: :class:`str`
                Name of PNG file
        :Effects:
            *db.png_fnames*: :class:`dict`
                Entry for *png* set to *fpng*
        :Versions:
            * 2020-03-31 ``@ddalle``: Version 1.0
        """
        # Check types
        if not typeutils.isstr(png):
            raise TypeError(
                "PNG name must be str (got %s)" % type(png))
        if not typeutils.isstr(fpng):
            raise TypeError(
                "png_name for '%s' must be str (got %s)"
                % (png, type(fpng)))
        # Check if file exists
        if not os.path.isfile(fpng):
            raise SystemError("No PNG file '%s' found" % fpng)
        # Ensure absolute file name
        fpng = os.path.abspath(fpng)
        # Get handle to attribute
        png_fnames = self.__dict__.setdefault("png_fnames", {})
        # Check type
        if not isinstance(png_fnames, dict):
            raise TypeError("png_fnames attribute is not a dict")
        # Set parameter (to a copy)
        png_fnames[png] = fpng

    # Set plot kwargs for named PNG
    def set_png_kwargs(self, png, **kw):
        r"""Set evaluation keyword arguments for PNG file

        :Call:
            >>> db.set_png_kwargs(png, kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *png*: :class:`str`
                Name used to tag this PNG image
            *kw*: {``{}``} | :class:`dict`
                Options to use when showing PNG image
        :Versions:
            * 2020-04-01 ``@jmeeroff``: Version 1.0
            * 2020-04-02 ``@ddalle``: Use :class:`MPLOpts`
            * 2020-05-26 ``@ddalle``: Combine existing *png_kwargs*
        """
        # Get handle to kw
        png_kwargs = self.__dict__.setdefault("png_kwargs", {})
        # Check types
        if not typeutils.isstr(png):
            raise TypeError(
                "PNG name must be str (got %s)" % type(png))
        elif not isinstance(png_kwargs, dict):
            raise TypeError("png_kwargs attribute is not a dict")
        # Get existing options
        kw_png = png_kwargs.get(png, {})
        # Combine options
        kw_png.update(**kw)
        # Convert to options and check
        kw = pmpl.MPLOpts(_sections=["imshow", "axformat"], **kw_png)
        # Save it
        png_kwargs[png] = kw

    # Add figure handle to list of figures
    def add_png_fig(self, png, fig):
        r"""Add figure handle to set of active figs for PNG tag

        :Call:
            >>> db.add_png_fig(png, fig)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *png*: :class:`str`
                Name/abbreviation/tag of PNG image to use
            *fig*: :class:`matplotlib.figure.Figure`
                Figure handle
        :Effects:
            *db.png_figs[png]*: :class:`set`
                Adds *fig* to :class:`set` if not already present
        :Versions:
            * 2020-04-01 ``@ddalle``: Version 1.0
        """
        # Get attribute
        png_figs = self.__dict__.setdefault("png_figs", {})
        # Get current handles
        figs = png_figs.setdefault(png, set())
        # Add it
        figs.add(fig)

    # Clear/reset list of figure handles
    def clear_png_fig(self, png):
        r"""Reset the set of figures for PNG tag

        :Call:
            >>> db.clear_png_fig(png)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *png*: :class:`str`
                Name/abbreviation/tag of PNG image to use
        :Effects:
            *db.png_figs[png]*: :class:`set`
                Cleared to empty :class:`set`
        :Versions:
            * 2020-04-01 ``@ddalle``: Version 1.0
        """
        # Get attribute
        png_figs = self.__dict__.setdefault("png_figs", {})
        # Get current handles
        figs = png_figs.setdefault(png, set())
        # Clear/reset it
        figs.clear()

   # --- Seam Curve: Get ---
    # Get plot kwargs for named seam
    def get_seam_kwargs(self, seam):
        r"""Set evaluation keyword arguments for PNG file

        :Call:
            >>> kw = db.set_seam_kwargs(seam)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *seam*: :class:`str`
                Name used to tag this seam curve
        :Outputs:
            *kw*: {``{}``} | :class:`MPLOpts`
                Options to use when showing seam curve (copied)
        :Versions:
            * 2020-04-03 ``@jmeeroff``: Version 1.0
        """
        # Get handle to kw
        seam_kwargs = self.__dict__.setdefault("seam_kwargs", {})
        # Check types
        if not typeutils.isstr(seam):
            raise TypeError(
                "Seam curve name must be str (got %s)" % type(seam))
        # Get kwargs
        kw = seam_kwargs.get(seam, {})
        # Create a copy
        return copy.copy(kw)

    # Get seam tag to use for a given col
    def get_col_seam(self, col):
        r"""Get name/tag of seam curve to use when plotting *col*

        :Call:
            >>> png = db.get_col_seam(col)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Data column for to associate with *png*
        :Outputs:
            *seam*: :class:`str`
                Name used to tag seam curve
        :Versions:
            * 2020-04-03 ``@jmeeroff``: Version 1.0
        """
        # Check types
        if not typeutils.isstr(col):
            raise TypeError(
                "Data column must be str (got %s)" % type(col))
        # Get handle to attribute
        col_seams = self.__dict__.setdefault("col_seams", {})
        # Get PNG name
        return col_seams.get(col)

    # Get pair of columns used for seam curve
    def get_seam_col(self, seam):
        r"""Get column names that define named seam curve

        :Call:
            >>> xcol, ycol = db.get_seam_col(col)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *seam*: :class:`str`
                Name used to tag this seam curve
        :Outputs:
            *xcol*: :class:`str`
                Name of *col* for seam curve *x* coords
            *ycol*: :class:`str`
                Name of *col* for seam curve *y* coords
        :Versions:
            * 2020-03-31 ``@ddalle``: Version 1.0
        """
        # Check types
        if not typeutils.isstr(seam):
            raise TypeError(
                "Seam name must be str (got %s)" % type(seam))
        # Get handle to attribute
        seam_cols = self.__dict__.setdefault("seam_cols", {})
        # Get pair of columns
        return seam_cols.get(seam, (None, None))

   # --- Seam Curve: Set ---
    # Read seam curves
    def make_seam(self, seam, fseam, xcol, ycol, cols, **kw):
        r"""Define and read a seam curve

        :Call:
            >>> db.make_seam(seam, fseam, xcol, ycol, cols, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *seam*: :class:`str`
                Name used to tag this seam curve
            *fseam*: :class:`str`
                Name of seam curve file written by ``triload``
            *xcol*: :class:`str`
                Name of *col* for seam curve *x* coords
            *ycol*: :class:`str`
                Name of *col* for seam curve *y* coords
            *kw*: {``{}``} | :class:`dict`
                Options to use when plotting seam curve
        :See Also:
            * :func:`set_cols_seam`
            * :func:`set_seam_col`
            * :func:`set_seam_kwargs`
        :Versions:
            * 2020-04-03 ``@ddalle``: Version 1.0
        """
        # Read a text file
        if fseam is not None:
            self.read_textdata(
                fseam, NanDivider=True, cols=[xcol, ycol], save=False)
        # Save col names
        self.set_seam_col(seam, xcol, ycol)
        # Save the keyword args
        self.set_seam_kwargs(seam, **kw)
        # Save columns
        self.set_cols_seam(cols, seam)

    # Set seam curve to use for list of *cols*
    def set_cols_seam(self, cols, seam):
        r"""Set name/tag of seam curve for several data columns

        :Call:
            >>> db.set_cols_seam(cols, seam)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *cols*: :class:`list`\ [:class:`str`]
                Data column for to associate with *png*
            *seam*: :class:`str`
                Name/abbreviation/tag of seam curve to use
        :Effects:
            *db.col_seams*: :class:`dict`
                Entry for *col* in *cols* set to *seam*
        :Versions:
            * 2020-04-02 ``@jmeeroff``: Version 1.0
        """
        # Check input
        if not isinstance(cols, (list, tuple)):
            raise TypeError(
                "List of cols must be 'list' (got '%s')" % type(cols))
        # Check each col
        for (j, col) in enumerate(cols):
            if not typeutils.isstr(col):
                raise TypeError(
                    "col %i must be 'str' (got '%s')" % (j, type(col)))
        # Loop through columns
        for col in cols:
            # Call individual function
            self.set_col_seam(col, seam)

    # Set *col* to use named seam curve
    def set_col_seam(self, col, seam):
        r"""Set name/tag of seam curve to use when plotting *col*

        :Call:
            >>> db.set_col_seam(col, seam)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *col*: :class:`str`
                Data column for to associate with *png*
            *seam*: :class:`str`
                Name/abbreviation/tag of seam curve to use
        :Effects:
            *db.col_seams*: :class:`dict`
                Entry for *col* set to *png*
        :Versions:
            * 2020-04-02 ``@jmeeroff``: Version 1.0
        """
        # Check types
        if not typeutils.isstr(seam):
            raise TypeError(
                "seam curve name must be str (got %s)" % type(seam))
        if not typeutils.isstr(col):
            raise TypeError(
                "Data column must be str (got %s)" % type(col))
        # Get handle to attribute
        col_seams = self.__dict__.setdefault("col_seams", {})
        # Check type
        if not isinstance(col_seams, dict):
            raise TypeError("col_seams attribute is not a dict")
        # Set parameter (to a copy)
        col_seams[col] = seam

    # Set column name for seam
    def set_seam_col(self, seam, xcol, ycol):
        r"""Set column names that define named seam curve

        :Call:
            >>> db.set_seam_col(seam, xcol, ycol)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *seam*: :class:`str`
                Name used to tag this seam curve
            *xcol*: :class:`str`
                Name of *col* for seam curve *x* coords
            *ycol*: :class:`str`
                Name of *col* for seam curve *y* coords
        :Effects:
            *db.seam_cols*: :class:`dict`
                Entry for *seam* set to (*xcol*, *ycol*)
        :Versions:
            * 2020-03-31 ``@ddalle``: Version 1.0
        """
        # Check types
        if not typeutils.isstr(seam):
            raise TypeError(
                "Seam curve name must be str (got %s)" % type(seam))
        if not typeutils.isstr(xcol):
            raise TypeError(
                "seam_colx for '%s' must be str (got %s)"
                % (seam, type(xcol)))
        if not typeutils.isstr(ycol):
            raise TypeError(
                "seam_coly for '%s' must be str (got %s)"
                % (seam, type(ycol)))
        # Check if cols are present
        if xcol not in self:
            raise KeyError("Seam '%s' missing xcol '%s'" % (seam, xcol))
        if ycol not in self:
            raise KeyError("Seam '%s' missing ycol '%s'" % (seam, ycol))
        # Get handle to attribute
        seam_cols = self.__dict__.setdefault("seam_cols", {})
        # Check type
        if not isinstance(seam_cols, dict):
            raise TypeError("seam_cols attribute is not a dict")
        # Set parameter (to a copy)
        seam_cols[seam] = (xcol, ycol)

    # Set plot kwargs for named seam curve
    def set_seam_kwargs(self, seam, **kw):
        r"""Set evaluation keyword arguments for seam curve

        :Call:
            >>> db.set_seam_kwargs(seam, kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *seam*: :class:`str`
                Name used to tag this seam curve
            *kw*: {``{}``} | :class:`dict`
                Options to use when showing seam curve
        :Versions:
            * 2020-04-02 ``@jmeeroff``: Version 1.0
            * 2020-05-26 ``@ddalle``: Combine existing *png_kwargs*
        """
        # Get handle to kw
        seam_kwargs = self.__dict__.setdefault("seam_kwargs", {})
        # Check types
        if not typeutils.isstr(seam):
            raise TypeError(
                "seam name must be str (got %s)" % type(seam))
        elif not isinstance(seam_kwargs, dict):
            raise TypeError("seam_kwargs attribute is not a dict")
        # Check for existing *kwargs*
        kw_seam = seam_kwargs.get(seam, {})
        # Update them
        kw_seam.update(**kw)
        # Convert to options and check
        kw = pmpl.MPLOpts(_sections=["plot", "axformat", "seam"], **kw_seam)
        # Save it
        seam_kwargs[seam] = kw

    # Add figure handle to list of figures for named seam curve
    def add_seam_fig(self, seam, fig):
        r"""Add figure handle to set of active figs for seam curve tag

        :Call:
            >>> db.add_seam_fig(seam, fig)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *seam*: :class:`str`
                Name used to tag this seam curve
            *fig*: :class:`matplotlib.figure.Figure`
                Figure handle
        :Effects:
            *db.seam_figs[seam]*: :class:`set`
                Adds *fig* to :class:`set` if not already present
        :Versions:
            * 2020-04-01 ``@ddalle``: Version 1.0
        """
        # Get current handles
        figs = self.seam_figs.setdefault(seam, set())
        # Add it
        figs.add(seam)

    # Check figure handle if it's in current set
    def check_seam_fig(self, seam, fig):
        r"""Check if figure is in set of active figs for seam curve tag

        :Call:
            >>> q = db.check_seam_fig(seam, fig)
        :Inputs:
            *db*: :class:`DataKit`
                Database with scalar output functions
            *seam*: :class:`str`
                Name used to tag this seam curve
            *fig*: ``None`` | :class:`matplotlib.figure.Figure`
                Figure handle
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not *fig* is in *db.seam_figs[seam]*
        :Versions:
            * 2020-04-01 ``@ddalle``: Version 1.0
        """
        # Get attribute
        seam_figs = self.__dict__.get("seam_figs", {})
        # Get current handles
        figs = seam_figs.get(seam, set())
        # Check for figure
        return fig in figs

   # --- Docstrings ---
    # Document functions
    pmpl.MPLOpts._doc_keys_fn(plot, "plot", indent=12)
    pmpl.MPLOpts._doc_keys_fn(plot_contour, "contour", indent=12)
    pmpl.MPLOpts._doc_keys_fn(plot_linear, "plot_linear", indent=12)
    pmpl.MPLOpts._doc_keys_fn(plot_scalar, "plot", indent=12)
    pmpl.MPLOpts._doc_keys_fn(
        plot_contour, "axformat", fmt_key="axkeys", indent=12)
  # >

  # ===================
  # Regularization
  # ===================
  # <
   # --- RBF ---
    # Regularization using radial basis functions
    def regularize_by_rbf(self, cols, args=None, **kw):
        r"""Regularize col(s) to full-factorial matrix of several args

        The values of each *arg* to use for the full-factorial matrix
        are taken from the *db.bkpts* dictionary, usually generated by
        :func:`get_bkpts`.  The values in *db.bkpts*, however, can be
        set manually in order to interpolate the data onto a specific
        matrix of points.

        :Call:
            >>> db.regularize_by_rbf(cols=None, args=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with response toolkit
            *cols*: :class:`list`\ [:class:`str`]
                List of output data columns to regularize
            *args*: {``None``} | :class:`list`\ [:class:`str`]
                List of arguments; default from *db.response_args*
            *scol*: {``None``} | :class:`str` | :class:`list`
                Optional name of slicing col(s) for matrix
            *cocols*: {``None``} | :class:`list`\ [:class:`str`]
                Other dependent input cols; default from *db.bkpts*
            *function*: {``"cubic"``} | :class:`str`
                Basis function for :class:`scipy.interpolate.Rbf`
            *tol*: {``1e-4``}  | :class:`float`
                Default tolerance to use in combination with *slices*
            *tols*: {``{}``} | :class:`dict`
                Dictionary of specific tolerances for *cols**
            *translators*: :class:`dict`\ [:class:`str`]
                Alternate names; *col* -> *trans[col]*
            *prefix*: :class:`str` | :class:`dict`
                Universal prefix or *col*-specific prefixes
            *suffix*: :class:`str` | :class:`dict`
                Universal suffix or *col*-specific suffixes
        :Versions:
            * 2018-06-08 ``@ddalle``: Version 1.0
            * 2020-02-24 ``@ddalle``: Version 2.0
        """
       # --- Options ---
        # Get translators
        trans = kw.get("translators", {})
        prefix = kw.get("prefix")
        suffix = kw.get("suffix")
        # Overall mask
        mask = kw.get("mask")
        # Translator args
        tr_args = (trans, prefix, suffix)
       # --- Status Checks ---
        # Get break points
        bkpts = self.__dict__.get("bkpts")
        # Check
        if bkpts is None:
            raise AttributeError(
                "Break point dict must be present; see get_bkpts()")
       # --- Cols Check ---
        # Convert single column
        if typeutils.isstr(cols):
            cols = [cols]
        # Check columns
        if not isinstance(cols, list):
            raise TypeError(
                "Regularization cols must be list, got %s" % type(cols))
        # Number of cols
        ncols = len(cols)
        # Check for empty list
        if ncols == 0:
            raise IndexError("Col list is empty")
        # Check each column
        for (j, col) in enumerate(cols):
            # Check type
            if not typeutils.isstr(col):
                raise TypeError(
                    "Col %i must be str, got %s" % (j, type(col)))
            # Check availability
            if col not in self:
                raise KeyError("Col '%s' is not in database" % col)
            # Get data type
            dtype = self.get_col_dtype(col)
            # Ensure float
            if not (dtype.startswith("float") or dtype.startswith("complex")):
                raise TypeError(
                    "Nonnumeric dtype '%s' for col '%s'" % (dtype, col))
       # --- Args Check ---
        # Default input args
        if args is None:
            # Use args for last *col*
            args = self.get_response_args(col)
        # Backup input args
        if args is None:
            # Initialize list
            args = []
            # Loop through keys of *bkpts*
            # Note uncontrolled order
            for arg in bkpts:
                # Check if used as *col*
                if arg in cols:
                    continue
                # Get data type
                dtype = self.get_col_dtype(arg)
                # Ensure float
                if (dtype is not None) and not dtype.startswith("float"):
                    continue
                # Otherwise use it
                args.append(arg)
        # Checks
        if not isinstance(args, list):
            raise TypeError("Arg list must be 'list', got %s" % type(args))
        # Number of input args
        narg = len(args)
        # Check types
        for (j, arg) in enumerate(args):
            # Check type
            if not typeutils.isstr(arg):
                raise TypeError(
                    "Arg %i must be str, got %s" % (j, type(arg)))
            # Check presence
            if arg not in bkpts:
                raise KeyError("No break points for arg '%s'" % arg)
       # --- Slice Cols ---
        # Get optional slice column
        scol = kw.get("scol")
        # Check for list
        if isinstance(scol, list):
            # Get additional slice keys
            subcols = scol[1:]
            # Single slice key
            maincol = scol[0]
        elif scol is None:
            # No slices at all
            subcols = []
            maincol = None
        else:
            # No additional slice keys
            subcols = []
            maincol = scol
            # List of slice keys
            scol = [scol]
        # Remove slice keys from arg list to interpolants
        if scol is None:
            # No checks
            iargs = args
        else:
            # Check against *scol*
            iargs = [arg for arg in args if arg not in scol]
            # Save original values for *maincol*
            mainvals = self.get_values(maincol).copy()
       # --- Full-Factorial Matrix ---
        # Get full-factorial matrix at the current slice value
        X, slices = self.get_fullfactorial(scol=scol, cols=args)
        # Original values retained for creating masks during slices
        X0 = {}
        # Number of output points
        nX = X[args[0]].size
       # --- Regularization ---
        # Perform interpolations
        for col in cols:
            # Translate column name
            colreg = self._translate_colname(col, *tr_args)
            # Status update
            if kw.get("v"):
                print("  Regularizing col '%s' -> '%s'" % (col, colreg))
            # Check for slices
            if scol is None:
                # One interpolant
                f = self.genr8_rbf(col, args, **kw)
                # Create tuple of input arguments
                x = tuple(X[arg] for arg in args)
                # Evaluate RBF
                V = f(*x)
            else:
                # Number of slices
                nslice = slices[maincol].size
                # Initialize data
                V = np.zeros_like(X[maincol])
                # Convert slices to indices within *db*
                masks, _ = self.find(scol, mapped=True, mask=mask, **slices)
                # Loop through slices
                for i in range(nslice):
                    # Status update
                    if kw.get("v"):
                        # Get main key value
                        m = slices[maincol][i]
                        # Get value in fixed number of characters
                        sv = ("%6g" % m)[:6]
                        # In-place status update
                        sys.stdout.write("    Slice %s=%s (%i/%i)\r"
                            % (maincol, sv, i+1, nslice))
                        sys.stdout.flush()
                    # Initialize mask
                    J = np.ones(nX, dtype="bool")
                    # Loop through cols that define slice
                    for k in scol:
                        # Get value
                        vk = slices[k][i]
                        # Constrain
                        J = np.logical_and(J, X[k]==vk)
                    # Get indices of slice
                    I = np.where(J)[0]
                    # Create interpolant for fixed value of *skey*
                    f = self.genr8_rbf(col, iargs, I=masks[i], **kw)
                    # Create tuple of input arguments
                    x = tuple(X[k][I] for k in iargs)
                    # Evaluate coefficient
                    V[I] = f(*x)
                # Status update
                if kw.get("v"):
                    # Get main key value
                    m = slices[maincol][i]
                    # Get value in fixed number of characters
                    sv = ("%6g" % m)[:6]
                    # In-place status update
                    sys.stdout.write("%72s\r" % "")
                    sys.stdout.flush()
            # Save the values
            self.save_col(colreg, V)
       # --- New Arg Values ---
        # Save the lookup values
        for arg in args:
            # Translate column name
            argreg = self._translate_colname(arg, *tr_args)
            # Save original values
            #X0[arg] = self.get_all_values(arg)
            # Save values
            self.save_col(argreg, X[arg])
            # Check if new
            if argreg != arg:
                # Get previous definition
                defn = self.get_defn(arg)
                # Save a copy
                self.defns[argreg] = self._defncls(**defn)
                # Link break points
                bkpts[argreg] = bkpts[arg]
       # --- Co-mapped XAargs ---
        # Trajectory co-keys
        cocols = kw.get("cocols", list(bkpts.keys()))
        # Map other breakpoint keys
        for col in cocols:
            # Skip if already present
            if col in args:
                continue
            elif col in cols:
                continue
            # Check for slices
            if maincol is None:
                break
            # Translate col name
            colreg = self._translate_colname(col, *tr_args)
            # Get values for this column
            V0 = self.get_all_values(col)
            # Check if present
            if V0 is None:
                raise KeyError("No *cocol* called '%s' found" % col)
            # Check size
            if mainvals.size != V0.size:
                # Original sizes do not match; no map applicable
                continue
            # Regular matrix values of slice key
            M = X[maincol]
            # Initialize data
            V = np.zeros_like(M)
            # Initialize break points
            T = []
            # Status update
            if kw.get("v"):
                print("  Mapping key '%s'" % col)
            # Loop through slice values
            for m in bkpts[maincol]:
                # Find value of slice key matching that parameter
                i = np.where(mainvals == m)[0][0]
                # Output value
                v = V0[i]
                # Get the indices of break points with that value
                J = np.where(M == m)[0]
                # Evaluate coefficient
                V[J] = v
                # Save break point
                T.append(v)
            # Save the values
            self.save_col(colreg, V)
            # Save break points
            bkpts[colreg] = np.array(T)

   # --- Griddata ---
    # Regularize using piecewise linear
    def regularize_by_griddata(self, cols, args=None, **kw):
        r"""Regularize col(s) to full-factorial matrix of several args

        The values of each *arg* to use for the full-factorial matrix
        are taken from the *db.bkpts* dictionary, usually generated by
        :func:`get_bkpts`.  The values in *db.bkpts*, however, can be
        set manually in order to interpolate the data onto a specific
        matrix of points.

        :Call:
            >>> db.regularize_by_griddata(cols=None, args=None, **kw)
        :Inputs:
            *db*: :class:`DataKit`
                Database with response toolkit
            *cols*: :class:`list`\ [:class:`str`]
                List of output data columns to regularize
            *args*: {``None``} | :class:`list`\ [:class:`str`]
                List of arguments; default from *db.response_args*
            *scol*: {``None``} | :class:`str` | :class:`list`
                Optional name of slicing col(s) for matrix
            *cocols*: {``None``} | :class:`list`\ [:class:`str`]
                Other dependent input cols; default from *db.bkpts*
            *method*: {``"linear"``} | ``"cubic"`` | ``"nearest"``
                Interpolation method; ``"cubic"`` only for 1D or 2D
            *rescale*: ``True`` | {``False``}
                Rescale input points to unit cube before interpolation
            *tol*: {``1e-4``}  | :class:`float`
                Default tolerance to use in combination with *slices*
            *tols*: {``{}``} | :class:`dict`
                Dictionary of specific tolerances for single *cols*
            *translators*: :class:`dict`\ [:class:`str`]
                Alternate names; *col* -> *trans[col]*
            *prefix*: :class:`str` | :class:`dict`
                Universal prefix or *col*-specific prefixes
            *suffix*: :class:`str` | :class:`dict`
                Universal suffix or *col*-specific suffixes
        :Versions:
            * 2020-03-10 ``@ddalle``: Version 1.0
        """
       # --- Options ---
        # Get translators
        trans = kw.get("translators", {})
        prefix = kw.get("prefix")
        suffix = kw.get("suffix")
        # Overall mask
        mask = kw.get("mask")
        # Translator args
        tr_args = (trans, prefix, suffix)
       # --- Status Checks ---
        # Get break points
        bkpts = self.__dict__.get("bkpts")
        # Check
        if bkpts is None:
            raise AttributeError(
                "Break point dict must be present; see get_bkpts()")
       # --- Cols Check ---
        # Convert single column
        if typeutils.isstr(cols):
            cols = [cols]
        # Check columns
        if not isinstance(cols, list):
            raise TypeError(
                "Regularization cols must be list, got %s" % type(cols))
        # Number of cols
        ncols = len(cols)
        # Check for empty list
        if ncols == 0:
            raise IndexError("Col list is empty")
        # Check each column
        for (j, col) in enumerate(cols):
            # Check type
            if not typeutils.isstr(col):
                raise TypeError(
                    "Col %i must be str, got %s" % (j, type(col)))
            # Check availability
            if col not in self:
                raise KeyError("Col '%s' is not in database" % col)
            # Get data type
            dtype = self.get_col_dtype(col)
            # Ensure float
            if not (dtype.startswith("float") or dtype.startswith("complex")):
                raise TypeError(
                    "Nonnumeric dtype '%s' for col '%s'" % (dtype, col))
       # --- Args Check ---
        # Default input args
        if args is None:
            # Use args for last *col*
            args = self.get_response_args(col)
        # Backup input args
        if args is None:
            # Initialize list
            args = []
            # Loop through keys of *bkpts*
            # Note uncontrolled order
            for arg in bkpts:
                # Check if used as *col*
                if arg in cols:
                    continue
                # Get data type
                dtype = self.get_col_dtype(arg)
                # Ensure float
                if (dtype is not None) and not dtype.startswith("float"):
                    continue
                # Otherwise use it
                args.append(arg)
        # Checks
        if not isinstance(args, list):
            raise TypeError("Arg list must be 'list', got %s" % type(args))
        # Number of input args
        narg = len(args)
        # Check types
        for (j, arg) in enumerate(args):
            # Check type
            if not typeutils.isstr(arg):
                raise TypeError(
                    "Arg %i must be str, got %s" % (j, type(arg)))
            # Check presence
            if arg not in bkpts:
                raise KeyError("No break points for arg '%s'" % arg)
       # --- Slice Cols ---
        # Get optional slice column
        scol = kw.get("scol")
        # Check for list
        if isinstance(scol, list):
            # Get additional slice keys
            subcols = scol[1:]
            # Single slice key
            maincol = scol[0]
        elif scol is None:
            # No slices at all
            subcols = []
            maincol = None
        else:
            # No additional slice keys
            subcols = []
            maincol = scol
            # List of slice keys
            scol = [scol]
        # Remove slice keys from arg list to interpolants
        if scol is None:
            # No checks
            iargs = args
        else:
            # Check against *scol*
            iargs = [arg for arg in args if arg not in scol]
            # Save original values for *maincol*
            mainvals = self.get_values(maincol).copy()
       # --- Full-Factorial Matrix ---
        # Get full-factorial matrix at the current slice value
        X, slices = self.get_fullfactorial(scol=scol, cols=args)
        # Number of output points
        nX = X[args[0]].size
       # --- Regularization ---
        # Perform interpolations
        for col in cols:
            # Translate column name
            colreg = self._translate_colname(col, *tr_args)
            # Status update
            if kw.get("v"):
                print("  Regularizing col '%s' -> '%s'" % (col, colreg))
            # Check for slices
            if scol is None:
                # Create inputs
                x = tuple(X[k] for k in args)
                # Single grid weights
                W = self.genr8_griddata_weights(args, *x, **kw)
                # Reference values
                Y = self.get_values(col, mask)
                # Multiply weights
                V = np.dot(W, Y)
            else:
                # Number of slices
                nslice = slices[maincol].size
                # Number of output points
                nout = len(X[maincol])
                # Get initial values
                V0 = self.get_all_values(col)
                # Extra dimensions from inputs to be copied
                shape0 = V0.shape[:-1]
                # Number of dimensions
                ndim = V0.ndim
                # Initialize data
                V = np.zeros(shape0 + (nout,), dtype=V0.dtype)
                # Convert slices to indices within *db*
                masks, _ = self.find(scol, mapped=True, mask=mask, **slices)
                # Loop through slices
                for i in range(nslice):
                    # Status update
                    if kw.get("v"):
                        # Get main key value
                        m = slices[maincol][i]
                        # Get value in fixed number of characters
                        sv = ("%6g" % m)[:6]
                        # In-place status update
                        sys.stdout.write("    Slice %s=%s (%i/%i)\r"
                            % (maincol, sv, i+1, nslice))
                        sys.stdout.flush()
                    # Initialize mask
                    J = np.ones(nX, dtype="bool")
                    # Loop through cols that define slice
                    for k in scol:
                        # Get value
                        vk = slices[k][i]
                        # Constrain
                        J = np.logical_and(J, X[k]==vk)
                    # Get indices of slice
                    I = np.where(J)[0]
                    # Create tuple of input arguments test values
                    x = tuple(X[k][I] for k in iargs)
                    # Create interpolant for fixed value of *scol*
                    W = self.genr8_griddata_weights(
                        iargs, *x, I=masks[i], **kw)
                    # Get database values
                    Y = self.get_values(col, masks[i])
                    # Evaluate coefficient
                    if ndim == 1:
                        # Scalar
                        V[I] = np.dot(W, Y)
                    elif ndim == 2:
                        # Linear output
                        V[:,I] = np.dot(Y, W.T)
                # Clean up prompt
                if kw.get("v"):
                    sys.stdout.write("%72s\r" % "")
                    sys.stdout.flush()
            # Save the values
            self.save_col(colreg, V)
       # --- Co-mapped XAargs ---
        # Trajectory co-keys
        cocols = kw.get("cocols", list(bkpts.keys()))
        # Map other breakpoint keys
        for col in cocols:
            # Skip if already present
            if col in args:
                continue
            elif col in cols:
                continue
            # Check for slices
            if maincol is None:
                break
            # Translate col name
            colreg = self._translate_colname(col, *tr_args)
            # Get values for this column
            V0 = self.get_all_values(col)
            # Check size
            if mainvals.size != V0.size:
                # Original sizes do not match; no map applicable
                continue
            # Regular matrix values of slice key
            M = X[maincol]
            # Initialize data
            V = np.zeros_like(M)
            # Initialize break points
            T = []
            # Status update
            if kw.get("v"):
                print("  Mapping key '%s'" % col)
            # Loop through slice values
            for m in bkpts[maincol]:
                # Find value of slice key matching that parameter
                i = np.where(mainvals == m)[0][0]
                # Output value
                v = V0[i]
                # Get the indices of break points with that value
                J = np.where(M == m)[0]
                # Evaluate coefficient
                V[J] = v
                # Save break point
                T.append(v)
            # Save the values
            self.save_col(colreg, V)
            # Save break points
            bkpts[colreg] = np.array(T)
       # --- Regularized Arg Values ---
        # Save the lookup values
        for arg in args:
            # Translate column name
            argreg = self._translate_colname(arg, *tr_args)
            # Save values
            self.save_col(argreg, X[arg])
            # Check if new
            if argreg != arg:
                # Get previous definition
                defn = self.get_defn(arg)
                # Save a copy
                self.defns[argreg] = self._defncls(**defn)
                # Link break points
                bkpts[argreg] = bkpts[arg]
  # >


# Combine options
kwutils._combine_val(DataKit._tagmap, ftypes.BaseData._tagmap)
