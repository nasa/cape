# -*- coding: utf-8 -*-
r"""
:mod:`cape.dkit.tecdatfile`: ASCII Tecplot(R) column files
===================================================================

This module contains a basic interface in the spirit of
:mod:`cape.dkit.ftypes` for ASCII tecplot .dat files.  It
creates a class, :class:`TecDatFile` that does not rely on the popular
:func:`numpy.loadtxt` function.

If possible, the column names (which become keys in the
:class:`dict`-like class) are read from the header.  If the file
begins with multiple comment lines, the column names are read from the
final comment before the beginning of data.
"""

# Standard library
import re

# Third-party modules
import numpy as np

# Local imports
from .basefile import BaseFile, BaseFileDefn, BaseFileOpts, TextInterpreter
from ..tnakit import typeutils


# Regular expressions
REGEX_NUMERIC = re.compile(r"\d")
REGEX_ALPHA = re.compile("[A-z_]")


# Options
class TecDatFileOpts(BaseFileOpts):
    pass


# Definition
class TecDatFileDefn(BaseFileDefn):
    pass


# Options for TecDatFile.write_csv()
class _WriteTecDatOpts(TecDatFileOpts):
  # ===================
  # Class Attributes
  # ===================
  # <
   # --- Global Options ---
    # List of options
    _optlist = {
        "Comment",
        "Delimiter",
        "ExpChar",
        "ExpChars",
        "ExpMax",
        "ExpMaxs",
        "ExpMin",
        "ExpMins",
        "Precision",
        "Precisions",
        "Translators",
        "WriteFormat",
        "WriteFormats",
    }

    # Alternate names
    _optmap = {
        "comment": "Comment",
        "comments": "Comment",
        "delim": "Delimiter",
        "delimiter": "Delimiter",
        "echar": "ExpChar",
        "echars": "ExpChars",
        "emax": "ExpMax",
        "emaxs": "ExpMaxs",
        "emin": "ExpMin",
        "emins": "ExpMins",
        "format": "WriteFormat",
        "formats": "WriteFormats",
        "fmt": "WriteFormat",
        "fmts": "WriteFormats",
        "prec": "Precision",
        "precs": "Precisions",
        "precision": "Precision",
        "precisions": "Precisions",
    }

   # --- Types ---
    # Types allowed
    _opttypes = {
        "Comment": typeutils.strlike,
        "Delimiter": typeutils.strlike,
        "ExpChar": typeutils.strlike,
        "ExpChars": dict,
        "ExpMax": int,
        "ExpMaxs": dict,
        "ExpMin": int,
        "ExpMins": dict,
        "Precision": typeutils.intlike,
        "Precisions": dict,
        "WriteFormat": typeutils.strlike,
        "WriteFormats": dict,
    }

   # --- Defaults ---
    _rc = {
        "Comment": "#",
        "Delimeter": ", ",
        "ExpChar": "e",
        "ExpMax": 4,
        "ExpMin": -2,
        "Precision": 6,
    }
  # >


# Add definition support to option
TecDatFileOpts.set_defncls(TecDatFileDefn)
# Combine options
_WriteTecDatOpts.combine_optdefs()


# Class for handling data from Tec .dat files
class TecDatFile(BaseFile, TextInterpreter):
    r"""Class for reading Tecplot ASCII .dat files

    :Call:
        >>> db = TecDatFile(fname, **kw)
        >>> db = TecDatFile(f, **kw)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *f*: :class:`file`
            Open file handle
    :Outputs:
        *db*: :class:`cape.dkit.ftypes.tecdatfile.TecDatFile`
            CSV file interface
        *db.cols*: :class:`list`\ [:class:`str`]
            List of columns read
        *db.opts*: :class:`TecDatFileOpts`
            Options for this instance
        *db.defns*: :class:`dict`\ [:class:`TecDatFileDefn`]
            Definitions for each column
        *db[col]*: :class:`np.ndarray` | :class:`list`
            Numeric array or list of strings for each column
    :See also:
        * :class:`cape.dkit.ftypes.basefile.BaseFile`
        * :class:`cape.dkit.ftypes.basefile.TextInterpreter`
    :Versions:
        * 2023-08-30 ``@jmeeroff``: v1.0
    """
  # ==================
  # Class Attributes
  # ==================
  # <
   # --- Options ---
    # Class for options
    _optscls = TecDatFileOpts
    # Definition class
    _defncls = TecDatFileDefn
  # >

  # ======
  # Config
  # ======
  # <
    # Initialization method
    def __init__(self, fname=None, **kw):
        r"""Initialization method

        :Versions:
            * 2019-11-12 ``@ddalle``: v1.0
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
            self.read_tecdat(fname)
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
    def read_tecdat(self, fname):
        r"""Read a Tecplot ASCII .dat file, including header

        Reads either entire file or from current location

        :Call:
            >>> db.read_tecdat(f)
            >>> db.read_tecdate(fname)
        :Inputs:
            *db*: :class:`cape.dkit.ftypes.tecdatfile.TecDatFile`
                Tecplot ASCII .dat file interface
            *f*: :class:`file`
                File open for reading
            *fname*: :class:`str`
                Name of file to read
        :Versions:
            * 2019-11-25 ``@ddalle``: v1.0
        """
        # Check type
        if typeutils.isfile(fname):
            # Safe file name
            self.fname = fname.name
            # Already a file
            self._read_tecdat(fname)
        else:
            # Save file name
            self.fname = fname
            # Open file
            with open(fname, 'r') as f:
                # Process file handle
                self._read_tecdat(f)

    # Read CSV file from file handle
    def _read_tecdat(self, f):
        r"""Read a CSV file from current position

        :Call:
            >>> db._read_csv(f)
        :Inputs:
            *db*: :class:`cape.dkit.ftypes.csvfile.CSVFile`
                CSV file interface
            *f*: :class:`file`
                File open for reading
        :See Also:
            * :func:`read_csv_header`
            * :func:`read_csv_data`
        :Versions:
            * 2019-12-06 ``@ddalle``: v1.0
        """
        # Process title
        self.read_tecdat_title(f)
        # Process column names from "variables"
        self.read_tecdat_variables(f)
        # Process column types
        self.finish_defns()
        # Process zone
        self.read_tecdat_zone(f)
        # Loop through lines
        self.read_tecdat_data(f)

   # --- Header ---
    # Read title
    def read_tecdat_title(self, f):
        r"""Read ASCII tecplot .dat file title from first line of file

        :Call:
            >>> db.read_tecdat_title(f)
        :Inputs:
            *db*: :class:`cape.dkit.ftypes.tecdatfile.TecDatFile`
                Tecplot ASCII .dat file interface
            *f*: :class:`file`
                Open file handle
        :Effects:
            *db.title*: :class:`str`
                Title of .dat file
        :Versions:
            * 2019-11-12 ``@ddalle``: v1.0
        """
        # Read line
        line = f.readline()
        # Save Title
        self.title = line[7:-2]

    # Read variable line
    def read_tecdat_variables(self, f):
        r"""Read ASCII tecplot .dat file variables from second line of
         file

        :Call:
            >>> db.read_tecdat_variables(f)
        :Inputs:
            *db*: :class:`cape.dkit.ftypes.tecdatfile.TecDatFile`
                Tecplot ASCII .dat file interface
            *f*: :class:`file`
                Open file handle
        :Effects:
            *db.cols*: :class:`list`\ [:class:`str`]
                List of column names
        :Versions:
            * 2019-11-12 ``@ddalle``: v1.0
        """
        # Read line
        line = f.readline()
        # Get variable names
        rhs = line.split('=', 1)[1]
        cols = rhs.split(' ')
        breakpoint()
        # Save column names if reaching this point
        self.cols = self.translate_colnames(cols)
        # Output column names for kicks
        return cols

    # Read zone line
    def read_tecdat_zone(self, f):
        r"""Read ASCII tecplot .dat file zone from third line of
         file

        :Call:
            >>> db.read_tecdat_variables(f)
        :Inputs:
            *db*: :class:`cape.dkit.ftypes.tecdatfile.TecDatFile`
                Tecplot ASCII .dat file interface
            *f*: :class:`file`
                Open file handle
        :Effects:
            *db.t*: :class:`str`
                Title of zone
            *db.solutiontime*: :class:`float`
                Solution time of zone
            *db.strandid*: :class:`int`
                Strandid of zone
            *db.n*: :class:`int`
                Number of points in zone
            *db.f*: :class:`str`
                Format of zone
        :Versions:
            * 2019-11-12 ``@ddalle``: v1.0
        """
        # Read line
        line = f.readline()
        # Get variable names
        vals = line[5:-1].split(',')
        # Save zone values
        for i, val in enumerate(vals):
           # Split entry by the = sign
            item = val.split('=')
            # Get var name
            pointer = item[0].strip(" ")
            # Get var value
            v = item[-1].strip(" ")
            # Strip and quotation marks
            if v.startswith('"'):
                v = v.strip('"').strip(" ")
            # Convert to int if you can
            try:
                v = int(v)
                # Set the attribute
                self.__setattr__(pointer, v)
                continue
            except Exception:
                pass
            # Now try float
            try:
                v = float(v)
                # Set the attribute
                self.__setattr__(pointer, v)
                continue
            except Exception:
                pass
            # Set the attribute
            self.__setattr__(pointer, v)

   # --- Data ---
    # Read data
    def read_tecdat_data(self, f):
        r"""Read data portion of Tecplot ASCII dat file

        :Call:
            >>> db.read_tecdat_data(f)
        :Inputs:
            *db*: :class:`cape.dkit.ftypes.tecdatfile.TecDatFile`
                Tecplot ASCII file interface
            *f*: :class:`file`
                Open file handle
        :Effects:
            *db.cols*: :class:`list`\ [:class:`str`]
                List of column names
        :Versions:
            * 2023-08-25 ``@ddalle``: v1.0
        """
        # Loop through columns
        for col in self.cols:
            print(col)
            # Initialize  empty numpy array
            V = np.array([])
            # Loop until number of items reaced
            while np.size(V) < self.i:
                # Read line
                line = f.readline()
                # Get Values
                vals = line.strip().split()
                # Convert to float
                vals_f = [float(x) for x in vals]
                # Append to V
                V = np.append(V, vals_f)
                print(np.size(V))

            # Save col
            self.save_col(col, V)

  # >

  # =============
  # Write
  # =============
  # <
   # --- Write Drivers ---

  # >
# class CSVFile
