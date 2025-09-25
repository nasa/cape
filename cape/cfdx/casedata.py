r"""
:mod:`cape.cfdx.casedata`: Single-case data interfaces
=======================================================

This module provides the classes :class:`CaseFM` and
:class:`CaseResid` that resid iterative histories of various types.
"""

# Standard library
import os
from typing import Optional

# Third-party modules
import numpy as np

# Local imports
from .. import util
from ..dkit import capefile
from ..dkit.rdb import DataKit
from ..optdict import OptionsDict

# Module placeholder
plt = None


# Radian -> degree conversion
deg = np.pi / 180.0
DEG = deg


# Column names
CASE_COL_NAMES = "sourcefiles_list"
CASE_COL_MTIME = "sourcefiles_mtime"
CASE_COL_ITSRC = "iter_sourcefile"
CASE_COL_ITERS = "i"
CASE_COL_ITRAW = "solver_iter"
CASE_COL_TIME = "t"
CASE_COL_TRAW = "solver_time"
CASE_COL_PARENT = "col_parent"
CASE_COL_SUB_NAMES = "subiter_sourcefiles_list"
CASE_COL_SUB_MTIME = "subiter_sourcefiles_mtime"
CASE_COL_SUB_ITSRC = "subiter_sourcefile"
CASE_COL_SUB_ITERS = "i_sub"
CASE_COL_SUB_ITRAW = "solver_subiter"
CASE_COL_BASE_ITERS = "i_0"
CASE_COL_BASE_ITRAW = "solver_iter_0"
CASE_COL_BASE_ITSRC = "iter_0_sourcefile"
CASE_COL_PNAMES = "sourfefiles_parent_list"
CASE_COL_PMTIME = "sourcefiles_parent_mtime"
CASE_COL_PITERC = "sourcefiles_parent_iter"
CASEDATA_ITER_COLS = (
    CASE_COL_ITERS,
    CASE_COL_ITRAW,
    CASE_COL_ITSRC,
)
CASEDATA_SPECIAL_COLS = (
    CASE_COL_NAMES,
    CASE_COL_MTIME,
    CASE_COL_ITERS,
    CASE_COL_ITRAW,
    CASE_COL_ITSRC,
    CASE_COL_TIME,
    CASE_COL_TRAW,
    CASE_COL_PARENT,
    CASE_COL_SUB_NAMES,
    CASE_COL_SUB_MTIME,
    CASE_COL_SUB_ITERS,
    CASE_COL_SUB_ITRAW,
    CASE_COL_SUB_ITSRC,
    CASE_COL_BASE_ITERS,
    CASE_COL_BASE_ITRAW,
    CASE_COL_BASE_ITSRC,
)

# Constants
_FONT_FAMILY = [
    "DejaVu Sans",
    "Arial",
]
FONT_FAMILY = [
]


# Dedicated function to load Matplotlib only when needed.
def ImportPyPlot():
    r"""Import :mod:`matplotlib.pyplot` if not already loaded

    :Call:
        >>> ImportPyPlot()
    :Versions:
        * 2014-12-27 ``@ddalle``: v1.0
    """
    # Make global variables
    global plt
    global tform
    global Text
    # Check for PyPlot.
    try:
        plt.gcf
    except AttributeError:
        # Check compatibility of the environment
        if os.environ.get('DISPLAY') is None:
            # Use a special MPL backend to avoid need for DISPLAY
            import matplotlib
            matplotlib.use('Agg')
        # Load the modules.
        import matplotlib.pyplot as plt
        # Other modules
        import matplotlib.transforms as tform
        from matplotlib.text import Text


# Database plot options class using optdict
class DBPlotOpts(OptionsDict):
    # Attributes
    __slots__ = ()

    # Everything is a ring by default
    _optring = {
        "_default_": True,
    }

    # Aliases
    _optmap = {
        "c": "color",
        "ls": "linestyle",
        "lw": "linewidth",
        "mew": "markeredgewidth",
        "mfc": "markerfacecolor",
        "ms": "markersize",
    }

    # Defaults
    _rc = {
        "color": "k",
    }


# Individual case, individual component base class
class CaseData(DataKit):
    r"""Base class for case iterative histories

    :Call:
        >>> fm = CaseData()
    :Outputs:
        *fm*: :class:`cape.cfdx.databook.CaseData`
            Base iterative history class
    :Versions:
        * 2015-12-07 ``@ddalle``: v1.0
        * 2024-01-10 ``@ddalle``: v2.0
    """
   # --- Class attributes ---
    # Attributes
    __slots__ = (
        "coeffs",
        "iter_cache",
    )

    # Default column lists
    _base_cols = (
        CASE_COL_ITERS,
        CASE_COL_ITSRC,
    )
    _base_coeffs = ()
    _special_cols = tuple(CASEDATA_SPECIAL_COLS)
    # Whether separate subiteration data is expected
    _has_subiters = False

   # --- __dunder__ ---
    # Initialization method
    def __init__(self, meta: bool = False, **kw):
        r"""Initialization method

        :Versions:
            * 2015-12-07 ``@ddalle``: v1.0
            * 2024-01-10 ``@ddalle``: v2.0; empty
            * 2024-01-21 ``@ddalle``: v2.1; use ``_base_cols``
            * 2024-01-22 ``@ddalle``: v2.2; auto-cache
            * 2024-08-09 ``@ddalle``: v2.3; better iter trimming
        """
        # Parent initialization
        DataKit.__init__(self, **kw)
        # Initialize base cols
        self.init_empty()
        # Initialize source file metadata
        self.init_sourcefiles(meta=meta)
        # Read data if possible
        self.read()
        # Trim any repeat iters
        self.trim_iters()
        # Get state of cache
        i0 = self.iter_cache
        i1 = self.get_lastiter()
        # De-None
        i0 = 0 if i0 is None else i0
        i1 = 0 if i1 is None else i1

   # --- I/O ---
    # Initialize file attritubets
    def init_sourcefiles(self, meta: bool = False):
        r"""Initialize file name list and metadata

        :Call:
            >>> h.init_sourcefiles(meta=False)
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history instance
            *meta*: ``True`` | {``False``}
                Option to only read metadata (skip subiterations)
        :Versions:
            * 2024-01-22 ``@ddalle``: v1.0
            * 2024-02-21 ``@ddalle``: v1.1; add subiteration hooks
            * 2025-08-05 ``@ddalle``: v1.2; add *meta*
        """
        # Initialize iteration that's been cached
        self.iter_cache = 0
        # Initialize special columns
        self.save_col(CASE_COL_NAMES, [])
        self.save_col(CASE_COL_MTIME, {})
        self.save_col(CASE_COL_ITSRC, np.zeros(0, dtype="int32"))
        self.save_col(CASE_COL_PARENT, {})
        # Initialize subiterations
        if (not meta) and self._has_subiters:
            self.save_col(CASE_COL_SUB_NAMES, [])
            self.save_col(CASE_COL_SUB_MTIME, {})
            self.save_col(CASE_COL_SUB_ITSRC, np.zeros(0, dtype="int32"))

    # Read from all sources, cache plus new raw data
    def read(self):
        r"""Read iterative histroy from all sources

        This will first attempt to read the cached histroy from a
        ``.cdb`` file and then ready any raw solver output files as
        necessary.

        :Call:
            >>> h.read()
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history instance
        :Versions:
            * 2024-01-22 ``@ddalle``: v1.0
            * 2024-02-21 ``@ddalle``: v1.1; add subiteration hooks
        """
        # Read cache
        self.read_cdb()
        # Get list of files processed in *cdb* file
        sourcefiles_cdb = self.get(CASE_COL_NAMES, [])
        # Get list of file names to read
        sourcefiles = self.get_filelist()
        # Check for changes in sourcefile list
        for j, fname in enumerate(sourcefiles):
            # Check if already read
            if fname in sourcefiles_cdb:
                # Get index
                j_cdb = sourcefiles_cdb.index(fname)
                # Check for file getting renamed
                # This occurs e.g. in pyfun_hist.dat -> pyfun_hist.00.dat
                if j > j_cdb:
                    # Reinitialize w/o cdb file
                    self.clear()
                    self.init_empty()
        # Loop through source files, skipping if already in .cdb
        for fname in sourcefiles:
            # Otherwise read the file as normal
            self.process_sourcefile(fname)
        # Check for subiters
        if not self._has_subiters:
            return
        # Read subiters
        sourcefiles = self.get_subiter_filelist()
        # Loop through subiteration source files
        for fname in sourcefiles:
            self.process_subiter_sourcefile(fname)

    # Get list of file(s) to read
    def get_filelist(self) -> list:
        r"""Get ordered list of files to read to build iterative history

        :Call:
            >>> filelist = h.get_filelist()
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history instance
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to read
        :Versions:
            * 2024-01-22 ``@ddalle``: v1.0
        """
        # This is an abstract method of CaseData
        return []

    # Get list of subiteration file(s) to read
    def get_subiter_filelist(self) -> list:
        r"""Get ordered list of files to read to build subiter history

        :Call:
            >>> filelist = h.get_subiter_filelist()
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history instance
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to read
        :Versions:
            * 2024-02-21 ``@ddalle``: v1.0
        """
        # This is an abstract method of CaseData
        return []

    # Process data file
    def process_sourcefile(self, fname: str):
        r"""Read data from a file (if necessary)

        In most cases, developers will **NOT** nead to customize this
        function for each application or for each solver.

        :Call:
            >>> h.process_sourcefile(fname)
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Versions:
            * 2024-01-20 ``@ddalle``: v1.0
        """
        # Get list of files already read
        sourcefiles = self.get_values(CASE_COL_NAMES)
        # Ensure presence of file name list
        if sourcefiles is None:
            sourcefiles = []
            self.save_col(CASE_COL_NAMES, sourcefiles)
        # Check if file already deleted
        if not os.path.isfile(fname):
            return
        # Get modification times
        mtimes = self.get_values(CASE_COL_MTIME)
        # Ensure mod times are present
        if mtimes is None:
            mtimes = {}
            self.save_col(CASE_COL_MTIME, mtimes)
        # Get modification time of *ftime*
        mtime = os.path.getmtime(fname)
        # Check if this is the most recently read file, but already read
        if fname in sourcefiles:
            # If it's the last file processed, check if it's new
            mtime_cache = mtimes.get(fname)
            # Check if we can use *mtime*
            if mtime_cache is not None:
                # Check modification time
                if mtime_cache >= mtime:
                    # Already read!
                    return
            # Get iteration history and which file each iter came from
            i = self.get_values(CASE_COL_ITERS)
            isrc = self.get_values(CASE_COL_ITSRC)
            # Index of this file
            jsrc = sourcefiles.index(fname)
            # Clear out data from previous read(s)
            if isrc is not None and i is not None:
                # Only keep iters from previous files
                self.apply_mask(isrc != jsrc)
        else:
            # This will be a new file
            jsrc = len(sourcefiles)
        # Read the file
        data = self.readfile(fname)
        # Merge (add new field or append)
        self.append_casedata(data, jsrc)
        # Update metadata *after* successful read
        if fname not in sourcefiles:
            # Add to source file list
            sourcefiles.append(fname)
        # Update *modtime*, whether new or old file
        mtimes[fname] = mtime

    # Eliminate overwritten iterations (result of imperfect restart)
    def trim_iters(self):
        r"""Trim iters that are followed by later restart at lower iter

        If a previous case continues past its last restart, the history
        may contain some iterations that get overwritten during the next
        run.

        :Call:
            >>> fm.trim_iters()
        :Inputs:
            *fm*: :class:`CaseData`
                Single-case iterative history instance
        :Versions:
            * 2024-08-09 ``@ddalle``: v1.0
        """
        # Trim iterative history cols
        self._trim_iters_parent(CASE_COL_ITERS)
        # Trim subiterative history cols
        self._trim_iters_parent(CASE_COL_SUB_ITERS)

    # Eliminate overwritten iterations for one group
    def _trim_iters_parent(self, parent: str):
        r"""Trim iterations based on a single parent column

        :Call:
            >>> fm._trim_iters_parent(parent)
        :Inputs:
            *parent*: :class:`str`
                Name of parent column, usually ``"i"``
        """
        # Get iterative history
        iters = self.get(parent)
        # Skip if no iterations of this category
        if iters is None:
            return
        # Identify which iterations to keep
        mask = _mask_repeat_iters(iters)
        # Get parents
        parents = self.get(CASE_COL_PARENT, {})
        # Loop through cols
        for col in self.cols:
            # Skip CASE_COL_PARENT
            if col == CASE_COL_PARENT:
                continue
            # Get value
            v = self[col]
            # Get parent
            col_parent = parents.get(col, CASE_COL_ITERS)
            # Check criteria
            if (
                    not isinstance(v, np.ndarray) or
                    col_parent != parent or
                    v.size != mask.size):
                continue
            # Save trimmed version
            self[col] = v[mask]

    # Process subiteration data file
    def process_subiter_sourcefile(self, fname: str):
        r"""Read data from a subiteration history file (if necessary)

        In most cases, developers will **NOT** nead to customize this
        function for each application or for each solver.

        :Call:
            >>> h.process_sourcefile(fname)
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Versions:
            * 2024-02-21 ``@ddalle``: v1.0
        """
        # Get list of files already read
        sourcefiles = self.get_values(CASE_COL_SUB_NAMES)
        # Ensure presence of file name list
        if sourcefiles is None:
            sourcefiles = []
            self.save_col(CASE_COL_SUB_NAMES, sourcefiles)
        # Check if file already deleted
        if not os.path.isfile(fname):
            return
        # Get modification times
        mtimes = self.get_values(CASE_COL_SUB_MTIME)
        # Ensure mod times are present
        if mtimes is None:
            mtimes = {}
            self.save_col(CASE_COL_SUB_MTIME, mtimes)
        # Get modification time of *ftime*
        mtime = os.path.getmtime(fname)
        # Check if this is the most recently read file, but already read
        if fname in sourcefiles:
            # If it's the last file processed, check if it's new
            mtime_cache = mtimes.get(fname)
            # Check if we can use *mtime*
            if mtime_cache is not None:
                # Check modification time
                if mtime_cache >= mtime:
                    # Already read!
                    return
            # Get iteration history and which file each iter came from
            i = self.get_values(CASE_COL_SUB_ITERS)
            isrc = self.get_values(CASE_COL_SUB_ITSRC)
            # Index of this file
            jsrc = len(sourcefiles) - 1
            # Clear out data from previous read(s)
            if isrc is not None and i is not None:
                # Only keep iters from previous files
                self.apply_mask(isrc != jsrc, parent=CASE_COL_SUB_ITERS)
        else:
            # This will be a new file
            jsrc = len(sourcefiles)
        # Read the file
        data = self.readfile_subiter(fname)
        # Merge (add new field or append)
        self.append_casedata(data, jsrc, typ="sub")
        # Generate data for the base of each major iter
        dbbase = self.genr8_subiter_base(data)
        # Save that, too
        self.append_casedata(dbbase, jsrc, typ="base")
        # Update metadata *after* successful read
        if fname not in sourcefiles:
            # Add to source file list
            sourcefiles.append(fname)
        # Update *modtime*, whether new or old file
        mtimes[fname] = mtime

    # Read data from a file
    def readfile(self, fname: str) -> dict:
        r"""Read raw data solver file and return a dict

        This method needs to be customized for each individual solver.

        :Call:
            >>> data = h.readfile(fname)
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *data*: :class:`dict`
                Data to add to or append to keys of *h*
        :Versions:
            * 2024-01-22 ``@ddalle``: v1.0
        """
        return {}

    # Read data from a subiteration file
    def readfile_subiter(self, fname: str) -> dict:
        r"""Read raw data subiteration solver file and return a dict

        This method needs to be customized for each individual solver.

        :Call:
            >>> data = h.readfile_subiter(fname)
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *data*: :class:`dict`
                Data to add to or append to keys of *h*
        :Versions:
            * 2024-02-21 ``@ddalle``: v1.0
        """
        return {}

    # Process "base" for each major iteration from subiteration histories
    def genr8_subiter_base(self, dbsub: dict) -> dict:
        r"""Sample first subiteration from each major iteration

        :Call:
            >>> data = h.genr8_subiter_base(dbsub)
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history instance
            *dbsub*: :class:`dict`
                Name of file to read
        :Outputs:
            *data*: :class:`dict`
                Data to add to or append to keys of *h*
        :Versions:
            * 2025-02-21 ``@ddalle``: v1.0
        """
        # Modify iteration to global history value
        i_sub = dbsub.get(CASE_COL_SUB_ITERS)
        # Ensure it's present
        if i_sub is None:
            raise KeyError(
                f"Missing required array-like col '{CASE_COL_SUB_ITERS}'")
        # Find indices of first subiteration at each major iteration
        mask0 = (i_sub == np.floor(i_sub))
        # Initialize output
        data = {}
        # Loop through cols
        for col, v in dbsub.items():
            # Create new column marking beginning of major iter
            if col == CASE_COL_SUB_ITERS:
                # Main iteration
                col0 = CASE_COL_BASE_ITERS
            elif col == CASE_COL_SUB_ITRAW:
                # Raw iteration reported by solver
                col0 = CASE_COL_BASE_ITRAW
            elif col == CASE_COL_SUB_ITSRC:
                # Index of source file
                col0 = CASE_COL_BASE_ITSRC
            else:
                # Replace suffix
                col0 = col.replace("_sub", "_0")
            # Save
            data[col0] = v[mask0]
        # Output
        return data

    # Get last iteration from a file
    def readfile_lastiter(self, fname: str) -> float:
        r"""Estimate the last iteration of a data file

        The purpose of this function is to determine if the file *fname*
        needs to be read. If negative, the file is always read.

        This function should be customized for each subclass. However,
        if it isn't, that just means the latest raw data file written by
        the solver is always read.

        :Call:
            >>> i = h.readfile_lastiter(fname)
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history index
            *fname*: :class:`str`
                Name of file to process
        :Outputs:
            *i*: {``-1.0``} | :class:`float`
                Laster iteration if *fname*
        """
        return -1.0

    # Get path to cache file
    def get_cdbfile(self) -> str:
        r"""Get path to iterative history cache file

        :Call:
            >>> cdbfilename = h.get_cdbfile()
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history instance
        :Outputs:
            *cdbfilename*: :class:`str`
                Name of file (extension is ``.cdb``)
        :Versions:
            * 2024-01-22 ``@ddalle``: v1.0
        """
        # Get file name
        return os.path.join("cape", "CASEDATA_CACHE.cdb")

    # Write to cached file
    def write_cdb(self):
        r"""Write contents of history to ``.cdb`` file

        See :mod:`capefile` module. The name of the file will be
        ``f"cape/fm_{fm.comp}.cdb"``.

        :Call:
            >>> fm.write_cdb()
        :Inputs:
            *fm*: :class:`CaseData`
                Iterative history instance
        :Versions:
            * 2024-01-20 ``@ddalle``: v1.0
        """
        # Get file name
        fname = self.get_cdbfile()
        # Try to write it
        try:
            # Create folder if necessary
            if not os.path.isdir("cape"):
                os.mkdir("cape")
            # Create database
            db = capefile.CapeFile(self)
            # Write file
            db.write(fname)
        except PermissionError:
            print(f"    Lacking permissions to write '{fname}'")

    # Read from cache
    def read_cdb(self):
        r"""Read contents of history from ``.cdb`` file

        See :mod:`capefile` module. The name of the file will be
        ``f"cape/fm_{fm.comp}.cdb"``.

        :Call:
            >>> fm.read_cdb()
        :Inputs:
            *fm*: :class:`CaseData`
                Iterative history instance
        :Versions:
            * 2024-01-20 ``@ddalle``: v1.0
            * 2024-01-22 ``@ddalle``: v1.1; _special_cols check
        """
        # Get file name
        fname = self.get_cdbfile()
        # Check for file name
        if os.path.isfile(fname):
            # Read it
            try:
                self._read_cdb(fname)
            except Exception:
                self.init_empty()
        # Mark iteration that was cached
        self.iter_cache = self.get_lastiter()

    def _read_cdb(self, fname: str):
        # Get class handle
        cls = self.__class__
        # Read it
        db = capefile.CapeFile(fname)
        # Store values
        for col in db.cols:
            # Save the data
            if col not in cls._special_cols:
                # Save as a "coeff"
                self.save_coeff(col, db[col])
            else:
                # Save as DataKit col but not iterative history
                self.save_col(col, db[col])

   # --- Iteration search ---
    # Get the current last iter
    def get_lastiter(self) -> float:
        r"""Get the last iteration saved to history

        :Call:
            >>> i = h.get_lastiter()
        :Inputs:
            *h*: :class:`CaseData`
                Individual-case iterative history instance
        :Outputs:
            *i*: :class:`float` | :class:`int`
                Laster iteration in *h*
        :Versions:
            * 2024-01-22 ``@ddalle``: v1.0
        """
        # Get iterations
        i = self.get(CASE_COL_ITERS)
        # Check for null
        if (i is None) or (i.size == 0):
            return 0.0
        else:
            return i[-1]

    # Get the current last iter
    def get_lastrawiter(self) -> float:
        r"""Get the last iteration saved to history

        :Call:
            >>> i = h.get_lastiter()
        :Inputs:
            *h*: :class:`CaseData`
                Individual-case iterative history instance
        :Outputs:
            *i*: :class:`float` | :class:`int`
                Laster iteration in *h*
        :Versions:
            * 2024-01-23 ``@ddalle``: v1.0
        """
        # Get iterations
        i = self.get(CASE_COL_ITRAW)
        # Check for null
        if (i is None) or (i.size == 0):
            return 0.0
        else:
            return i[-1]

    # Get the current last time
    def get_lasttime(self) -> float:
        r"""Get the last time step saved to history

        :Call:
            >>> t = h.get_lasttime()
        :Inputs:
            *h*: :class:`CaseData`
                Individual-case iterative history instance
        :Outputs:
            *t*: :class:`float`
                Laster time step in *t*, ``0.0`` if no time steps
        :Versions:
            * 2024-01-23 ``@ddalle``: v1.0
        """
        # Get iterations
        t = self.get(CASE_COL_TIME)
        # Check for null
        if (t is None) or (t.size == 0):
            # No time history
            return -1.0
        else:
            return t[-1]

    # Get the current last time
    def get_maxtime(self) -> float:
        r"""Get the last time step saved to history

        :Call:
            >>> t = h.get_maxtime()
        :Inputs:
            *h*: :class:`CaseData`
                Individual-case iterative history instance
        :Outputs:
            *t*: :class:`float`
                Laster time step in *t*, ``0.0`` if no time steps
        :Versions:
            * 2024-01-23 ``@ddalle``: v1.0
        """
        # Get iterations
        t = self.get(CASE_COL_TIME)
        # Check for null
        if (t is None) or (t.size == 0):
            # No time history
            return -1.0
        else:
            return np.max(t)

    # Function to get index of a certain iteration number
    def GetIterationIndex(self, i: int):
        r"""Return index of a particular iteration in *fm.i*

        If the iteration *i* is not present in the history, the index of the
        last available iteration less than or equal to *i* is returned.

        :Call:
            >>> j = fm.GetIterationIndex(i)
        :Inputs:
            *fm*: :class:`cape.cfdx.databook.CaseData`
                Case component history class
            *i*: :class:`int`
                Iteration number
        :Outputs:
            *j*: :class:`int`
                Index of last iteration less than or equal to *i*
        :Versions:
            * 2015-03-06 ``@ddalle``: v1.0 (``CaseFM``)
            * 2015-12-07 ``@ddalle``: v1.0
            * 2024-01-11 ``@ddalle``: v1.1; use keys instead of attrs
        """
        # Get iterations
        iters = self.get_all_values(CASE_COL_ITERS)
        # Check for *i* less than first iteration
        if iters.size == 0 or i < iters[0]:
            return 0
        # Find the index
        j = np.where(iters <= i)[0][-1]
        # Output
        return j

   # --- Data ---
    # Function to make empty one.
    def init_empty(self):
        r"""Create empty *CaseFM* instance

        :Call:
            >>> h.init_empty()
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history index
        :Versions:
            * 2015-10-16 ``@ddalle``: v1.0
            * 2023-01-11 ``@ddalle``: v2.0; DataKit updates
            * 2024-01-22 ``@ddalle``: v2.1; use class attributes
        """
        # Get class
        cls = self.__class__
        # Initialize all the columns
        for col in cls._base_cols:
            self.save_col(col, np.zeros(0))
        # Intitialize "coefficients"
        self.coeffs = list(cls._base_coeffs)

    # Extract one value/coefficient/state
    def ExtractValue(self, c: str, col=None, **kw):
        r"""Extract the iterative history for one coefficient/state

        This function may be customized for some modules

        :Call:
            >>> C = fm.Extractvalue(c)
            >>> C = fm.ExtractValue(c, col=None)
        :Inputs:
            *fm*: :class:`cape.cfdx.databook.CaseData`
                Case component history class
            *c*: :class:`str`
                Name of state
            *col*: {``None``} | :class:`int`
                Column number
        :Outputs:
            *C*: :class:`np.ndarray`
                Values for *c* at each iteration or sample interval
        :Versions:
            * 2015-12-07 ``@ddalle``: v1.0
            * 2024-01-10 ``@ddalle``: v2.0, CaseFM -> DataKit
        """
        # Get values directly
        v = self.get_values(c)
        # Check for special cases
        if v is None:
            if c in ("CF", "CT"):
                # Force magnitude
                CA = self.get_values("CA")
                CY = self.get_values("CY")
                CN = self.get_values("CN")
                # Add them up
                v = np.sqrt(CA*CA + CY*CY + CN*CN)
            else:
                raise KeyError(f"No column called '{c}'")
        # Check for column
        if (col is not None) and isinstance(v, np.ndarray) and v.ndim > 1:
            v = v[:, col]
        # Output
        return v

    # Append data from a dict
    def append_casedata(self, data: dict, jsrc=None, typ="raw"):
        r"""Append data read from a single file

        :Call:
            >>> h.append_casedata(data, jsrc=None, typ="raw")
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history instance
            *data*: :class:`dict`
                Dictionary of data to append to *h*
            *jsrc*: {``None``} | :class:`int`
                Index of source file to save for each iteration
            *typ*: {``"raw"``} | ``"sub"`` | ``"base"``
                Type of iteration data being saved
        :Versions:
            * 2024-01-22 ``@ddalle``: v1.0
            * 2024-02-21 ``@ddalle``: v1.1; add *typ*
            * 2024-06-24 ``@ddalle``: v1.2; call trim_repeat_iters()
        """
        # Save iteration data
        self._save_iterdata(data, jsrc, typ=typ)
        # Get appropriate parent column name
        if typ == "sub":
            # Subiterations
            parent = CASE_COL_SUB_ITERS
        elif typ == "base":
            # Base of each subiteration cycle
            parent = CASE_COL_BASE_ITERS
        else:
            # Normal iterations
            parent = CASE_COL_ITERS
        # Get "parents" column dict
        parents = self.get(CASE_COL_PARENT, {})
        # Loop through data keys
        for k, v in data.items():
            # Ensure parent saved properly
            parents.setdefault(k, parent)
            # Skip special cols
            if k in CASEDATA_SPECIAL_COLS:
                continue
            # Otherwise append the data
            self._append_col(k, v)

    # Save iteration data
    def _save_iterdata(self, data: dict, jsrc=None, typ="raw"):
        r"""Save iteration data from raw *data* dict

        :Versions:
            * 2024-01-22 ``@ddalle``: v1.0
            * 2024-02-21 ``@ddalle``: v2.0; add *typ*
        """
        # Get column names based on type
        if typ == "sub":
            # Subiterations
            icol = CASE_COL_SUB_ITERS
            rcol = CASE_COL_SUB_ITRAW
            tcol = "__NONE__"
            scol = CASE_COL_SUB_ITSRC
            fcol = CASE_COL_SUB_NAMES
        elif typ == "base":
            # Base of each subiteration cycle
            icol = CASE_COL_BASE_ITERS
            rcol = CASE_COL_BASE_ITRAW
            tcol = "__NONE__"
            scol = CASE_COL_BASE_ITSRC
            fcol = CASE_COL_SUB_NAMES
        else:
            # Normal iterations
            icol = CASE_COL_ITERS
            rcol = CASE_COL_ITRAW
            tcol = CASE_COL_TIME
            scol = CASE_COL_ITSRC
            fcol = CASE_COL_NAMES
        # Default indices
        if jsrc is None:
            jsrc = len(self.get_values(fcol))
        # Get iterations
        inew = data.get(icol)
        tnew = data.get(tcol)
        # Cannot process w/o
        if not isinstance(inew, np.ndarray):
            raise TypeError(
                "Cannot process new data w/o array-type key " +
                f"'{icol}'; found '{type(inew).__name__}'")
        # Create array of source
        isrc = np.full(inew.shape[-1], jsrc, dtype="int32")
        # Save iterations and source
        self._append_col(icol, inew)
        self._append_col(scol, isrc)
        # Check for raw solver iterations
        iraw = data.get(rcol)
        # Save raw-solver iteration numbers
        if iraw is None:
            # Just save the actual iterations
            self._append_col(rcol, inew)
        else:
            # Save modified raw iters numbers
            self._append_col(rcol, iraw)
        # Check for time processing
        if tnew is None:
            return
        # Save times
        self._append_col(tcol, tnew)
        # Check for raw solver time
        traw = data.get(CASE_COL_TRAW)
        # Save raw-solver iteration numbers
        if traw is None:
            # Just save the actual iterations
            self._append_col(CASE_COL_TRAW, tnew)
        else:
            # Save modified raw iters numbers
            self._append_col(CASE_COL_TRAW, traw)

    # Append to one col
    def _append_col(self, col: str, v: np.ndarray):
        r"""Append data to a single column

        :Versions:
            * 2024-01-22 ``@ddalle``: v1.0
        """
        # Check if values are new
        if col not in self:
            # Just save a-new
            self.save_col(col, v)
            return
        # Append *v* to existing values
        self[col] = np.hstack((self[col], v))

    # Save data as a *coeff*
    def save_coeff(self, col: str, v):
        r"""Save data as a "coefficient"

        :Call:
            >>> h.save_coeff(col, v)
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history index
            *col*: :class:`str`
                Name of coefficient/column
            *v*: :class:`object`
                Any value
        :Versions:
            * 2024-01-02 ``@ddalle``: v1.0
        """
        # Check if *col* is a "special" coeff
        if col in CASEDATA_SPECIAL_COLS:
            raise ValueError(
                f"Special col name '{col}' is reserved " +
                "and cannot be added as a coeff")
        # Check if it's in the list of coefficients
        if col not in self.coeffs:
            self.coeffs.append(col)
        # Pass to save_col() method
        self.save_col(col, v)

    # Remove data
    def apply_mask(self, mask=None, parent: str = CASE_COL_ITERS):
        r"""Remove subset of iterative history

        :Call:
            >>> h.apply_mask(mask, parent="i")
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history instance
            *mask*: :class:`np.ndarray`\ [:class:`bool` | :class:`int`]
                Optional mask of which cases to *keep*
            *parent*: {``"i"``} | :class:`str`
                Name of iterations column
        :Versions:
            * 2024-01-22 ``@ddalle``: v1.0
            * 2024-02-20 ``@ddalle``: v1.1; add *parent*
        """
        # Check for null action
        if mask is None:
            return
        # Get list of columns
        cols = [
            CASE_COL_ITERS,
            CASE_COL_ITSRC,
            CASE_COL_ITRAW,
            CASE_COL_TIME,
            CASE_COL_TRAW,
        ] + self.coeffs
        # Get parent cols
        parents = self.get(CASE_COL_PARENT, {})
        # Loop through cols
        for j, col in enumerate(cols):
            # Default parent
            pdef = CASE_COL_SUB_ITERS
            pdef = pdef if col.endswith("_sub") else CASE_COL_ITERS
            # Get name of parent column
            parentj = parents.get(col, pdef)
            # Check if matching target parent
            if parent != parentj:
                continue
            # Get values
            vj = self.get(col)
            # Skip if not an array
            if not isinstance(vj, np.ndarray) or vj.size < mask.size:
                continue
            # Get size
            nj = vj.shape[-1]
            # Check size
            if nj != mask.size:
                # Cannot trim *col* due to mismatch
                n = mask.size
                print(
                    f"   Cannot trim col '{col}' with size {nj}; expected {n}")
                continue
            # Apply the mask; delete (and/or duplicate?) data
            self[col] = vj[mask]

    # Simple mask, iters only
    def apply_mask_iters(self, mask: np.ndarray):
        r"""Apply a mask to simple iteration history

        :Call:
            >>> db.apply_mask_iters(mask)
        :Inputs:
            *db*: :class:`cape.cfdx.casedata.CaseData`
                Single-case iterative history interface
            *mask*: :class:`np.ndarray`\ [:class:`int` | :class:`bool`]
                Mask of indices or booleans of which to keep
        :Versions:
            * 2025-07-31 ``@ddalle``: v1.0
        """
        # Find appropriate columns
        cols = self.get_cols_parent(CASE_COL_ITERS)
        # Apply mask to each
        for col in cols:
            self[col] = self[col][mask]

    # Find columns by parent
    def get_cols_parent(self, col: str) -> list:
        r"""Get columns with a given parent

        :Call:
            >>> cols = db.get_cols_by_parent(col)
        :Inputs:
            *db*: :class:`CaseData`
                Iterative history data instance
            *col*: :class:`str`
                Name of parent column to search for
        :Outputs:
            *cols*: :class:`list`\ [:class:`str`]
                List of columns with parent of *col*
        :Versions:
            * 2025-07-31 ``@ddalle``: v1.0
        """
        # Get list of parents
        parents = self.get(CASE_COL_PARENT, {})
        # Get size of that parent
        n = self[col].size
        # Find columns by parent
        cols = []
        # Loop through cols
        for colj, vj in self.items():
            # Get parent
            parent = parents.get(colj, colj)
            # Check
            if (parent == col) and (len(vj) == n):
                cols.append(colj)
        # Output
        return cols

   # --- Plot ---
    # Basic plotting function
    def PlotValue(self, c: str, col=None, n=None, **kw):
        r"""Plot an iterative history of some value named *c*

        :Call:
            >>> h = fm.PlotValue(c, n=None, **kw)
        :Inputs:
            *fm*: :class:`cape.cfdx.databook.CaseData`
                Case component history class
            *c*: :class:`str`
                Name of coefficient to plot, e.g. ``'CA'``
            *col*: :class:`str` | :class:`int` | ``None``
                Select a column by name or index
            *n*: :class:`int`
                Only show the last *n* iterations
            *nMin*: {``0``} | :class:`int`
                First iteration allowed for use in averaging
            *nAvg*, *nStats*: {``100``} | :class:`int`
                Use at least the last *nAvg* iterations to compute an average
            *dnAvg*, *dnStats*: {*nStats*} | :class:`int`
                Use intervals of *dnStats* iterations for candidate windows
            *nMax*, *nMaxStats*: {*nStats*} | :class:`int`
                Use at most *nMax* iterations
            *d*: :class:`float`
                Delta in the coefficient to show expected range
            *k*: :class:`float`
                Multiple of iterative standard deviation to plot
            *u*: :class:`float`
                Multiple of sampling error standard deviation to plot
            *err*: :class:`float`
                Fixed sampling error, def uses :func:`util.SearchSinusoidFit`
            *nLast*: :class:`int`
                Last iteration to use (defaults to last iteration available)
            *nFirst*: :class:`int`
                First iteration to plot
            *FigureWidth*: :class:`float`
                Figure width
            *FigureHeight*: :class:`float`
                Figure height
            *PlotOptions*: :class:`dict`
                Dictionary of additional options for line plot
            *StDevOptions*: :class:`dict`
                Options passed to :func:`plt.fill_between` for stdev plot
            *ErrPltOptions*: :class:`dict`
                Options passed to :func:`plt.fill_between` for uncertainty plot
            *DeltaOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for reference range plot
            *MeanOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for mean line
            *ShowMu*: :class:`bool`
                Option to print value of mean
            *ShowSigma*: :class:`bool`
                Option to print value of standard deviation
            *ShowError*: :class:`bool`
                Option to print value of sampling error
            *ShowDelta*: :class:`bool`
                Option to print reference value
            *MuFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the mean value
            *DeltaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the reference value *d*
            *SigmaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the iterative standard deviation
            *ErrorFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the sampling error
            *XLabel*: :class:`str`
                Specified label for *x*-axis, default is ``I"teration Number"``
            *YLabel*: :class:`str`
                Specified label for *y*-axis, default is *c*
            *Grid*: {``None``} | ``True`` | ``False``
                Turn on/off major grid lines, or leave as is if ``None``
            *GridStyle*: {``{}``} | :class:`dict`
                Dictionary of major grid line line style options
            *MinorGrid*: {``None``} | ``True`` | ``False``
                Turn on/off minor grid lines, or leave as is if ``None``
            *MinorGridStyle*: {``{}``} | :class:`dict`
                Dictionary of minor grid line line style options
            *Ticks*: {``None``} | ``False``
                Turn off ticks if ``False``
            *XTicks*: {*Ticks*} | ``None`` | ``False`` | :class:`list`
                x-axis tick levels, turn off if ``False`` or ``[]``
            *YTicks*: {*Ticks*} | ``None`` | ``False`` | :class:`list`
                y-axis tick levels, turn off if ``False`` or ``[]``
            *TickLabels*: {``None``} | ``False``
                Turn off tick labels if ``False``
            *XTickLabels*:{``None``} | ``False`` | :class:`list`
                x-axis tick labels, turn off if ``False`` or ``[]``
            *YTickLabels*: {``None``} | ``False`` | :class:`list`
                y-axis tick labels, turn off if ``False`` or ``[]``
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: v1.0
            * 2014-12-09 ``@ddalle``: v1.1; move to ``AeroPlot`` class
            * 2015-02-15 ``@ddalle``: v1.2; move to ``Aero`` class
            * 2015-03-04 ``@ddalle``: v1.3; add *nStart* and *nLast*
            * 2015-12-07 ``@ddalle``: v1.4; move to ``CaseData``
            * 2017-10-12 ``@ddalle``: v1.5; add grid and tick options
            * 2024-01-10 ``@ddalle``: v1.6; DataKit updates
        """
       # ----------------
       # Initial Options
       # ----------------
        # Make sure plotting modules are present.
        ImportPyPlot()
        # Extract the data.
        if col:
            # Extract data with a separate column reference
            C = self.ExtractValue(c, col)
        else:
            # Extract from whole data set
            C = self.ExtractValue(c)
        # Process inputs.
        nLast = kw.get('nLast')
        nFirst = kw.get('nFirst', 1)
        # De-None
        if nFirst is None:
            nFirst = 1
        # Check if *nFirst* is negative
        if nFirst < 0:
            nFirst = self[CASE_COL_ITERS][-1] + nFirst
        # Iterative uncertainty options
        dc = kw.get("d", 0.0)
        ksig = kw.get("k", 0.0)
        uerr = kw.get("u", 0.0)
        # Other plot options
        fw = kw.get('FigureWidth')
        fh = kw.get('FigureHeight')
        # Get iterations
        iters = self.get_all_values("i")
       # ------------
       # Statistics
       # ------------
        # Averaging window size (minimum)
        nAvg  = kw.get("nAvg", kw.get("nStats", 100))
        # Increment in candidate window size
        dnAvg = kw.get("dnAvg", kw.get("dnStats", nAvg))
        # Maximum window size
        nMax = kw.get("nMax", kw.get("nMaxStats", nAvg))
        # Minimum allowed iteration
        nMin = kw.get("nMin", nFirst)
        # Get statistics
        s = util.SearchSinusoidFitRange(
            iters, C, nAvg, nMax,
            dn=dnAvg, nMin=nMin)
        # New averaging iteration
        nAvg = s['n']
       # ---------
       # Last Iter
       # ---------
        # Most likely last iteration
        iB = iters[-1]
        # Check for an input last iter
        if nLast is not None:
            # Attempt to use requested iter.
            if nLast < iB:
                # Using an earlier iter; make sure to use one in the hist.
                # Find the iterations that are less than i.
                jB = self.GetIterationIndex(nLast)
                iB = iters[jB]
        # Get the index of *iB* in *self.i*.
        jB = self.GetIterationIndex(iB)
       # ----------
       # First Iter
       # ----------
        # Don't cut off the entire history
        if nFirst >= iB:
            nFirst = 1
        # Default number of iterations: all
        if n is None:
            n = len(iters)
        j0 = max(0, jB - n)
        # Get the starting iteration number to use.
        i0 = max(0, iters[j0], nFirst) + 1
        # Make sure *iA* is in *iters* and get the index.
        j0 = self.GetIterationIndex(i0)
        # Reselect *i0* in case initial value was not in *self.i*.
        i0 = iters[j0]
       # --------------
       # Averaging Iter
       # --------------
        # Get the first iteration to use in averaging.
        jA = max(j0, jB-nAvg+1)
        # Reselect *iV* in case initial value was not in *self.i*.
        iA = iters[jA]
       # -----------------------
       # Standard deviation plot
       # -----------------------
        # Initialize dictionary of handles.
        h = {}
        # Shortcut for the mean
        cAvg = s['mu']
        # Initialize plot options for standard deviation
        kw_s = DBPlotOpts(
            color='b', lw=0.0,
            facecolor="b", alpha=0.35, zorder=1)
        # Calculate standard deviation if necessary
        if (ksig and nAvg > 2) or kw.get("ShowSigma"):
            c_std = s['sig']
        # Show iterative n*standard deviation
        if ksig and nAvg > 2:
            # Extract plot options from kwargs
            for k in util.denone(kw.get("StDevOptions", {})):
                # Ignore linestyle and ls
                if k in ('ls', 'linestyle'):
                    continue
                # Override the default option.
                if kw["StDevOptions"][k] is not None:
                    kw_s[k] = kw["StDevOptions"][k]
            # Limits
            cMin = cAvg - ksig*c_std
            cMax = cAvg + ksig*c_std
            # Plot the target window boundaries.
            h['std'] = plt.fill_between([iA, iB], [cMin]*2, [cMax]*2, **kw_s)
       # --------------------------
       # Iterative uncertainty plot
       # --------------------------
        kw_u = DBPlotOpts(
            color='g', lw=0,
            facecolor="g", alpha=0.35, zorder=2)
        # Calculate sampling error if necessary
        if (uerr and nAvg > 2) or kw.get("ShowError"):
            # Check for sampling error
            c_err = kw.get('err', s['u'])
        # Show iterative n*standard deviation
        if uerr and nAvg > 2:
            # Extract plot options from kwargs
            for k in util.denone(kw.get("ErrPltOptions", {})):
                # Ignore linestyle and ls
                if k in ('ls', 'linestyle'):
                    continue
                # Override the default option.
                if kw["ErrPltOptions"][k] is not None:
                    kw_u[k] = kw["ErrPltOptions"][k]
            # Limits
            cMin = cAvg - uerr*c_err
            cMax = cAvg + uerr*c_err
            # Plot the target window boundaries.
            h['err'] = plt.fill_between([iA, iB], [cMin]*2, [cMax]*2, **kw_u)
       # ---------
       # Mean plot
       # ---------
        # Initialize plot options for mean.
        kw_m = DBPlotOpts(
            color=kw.get("color", "0.1"),
            ls=[":", "-"], lw=1.0, zorder=8)
        # Extract plot options from kwargs
        for k in util.denone(kw.get("MeanOptions", {})):
            # Override the default option.
            if kw["MeanOptions"][k] is not None:
                kw_m[k] = kw["MeanOptions"][k]
        # Turn into two groups.
        kw0 = {}
        kw1 = {}
        for k in kw_m:
            kw0[k] = kw_m.get_opt(k, 0)
            kw1[k] = kw_m.get_opt(k, 1)
        # Plot the mean.
        h['mean'] = (
            plt.plot([i0, iA], [cAvg, cAvg], **kw0) +
            plt.plot([iA, iB], [cAvg, cAvg], **kw1))
       # ----------
       # Delta plot
       # ----------
        # Initialize options for delta
        kw_d = DBPlotOpts(color="r", ls="--", lw=0.8, zorder=4)
        # Calculate range of interest.
        if dc:
            # Extract plot options from kwargs
            for k in util.denone(kw.get("DeltaOptions", {})):
                # Override the default option.
                if kw["DeltaOptions"][k] is not None:
                    kw_d[k] = kw["DeltaOptions"][k]
            # Turn into two groups.
            kw0 = {}
            kw1 = {}
            for k in kw_m:
                kw0[k] = kw_d.get_opt(k, 0)
                kw1[k] = kw_d.get_opt(k, 1)
            # Limits
            cMin = cAvg-dc
            cMax = cAvg+dc
            # Plot the target window boundaries.
            h['min'] = (
                plt.plot([i0, iA], [cMin, cMin], **kw0) +
                plt.plot([iA, iB], [cMin, cMin], **kw1))
            h['max'] = (
                plt.plot([i0, iA], [cMax, cMax], **kw0) +
                plt.plot([iA, iB], [cMax, cMax], **kw1))
       # ------------
       # Primary plot
       # ------------
        # Initialize primary plot options.
        kw_p = DBPlotOpts(
            color=kw.get("color", "k"), ls="-", lw=1.5, zorder=7)
        # Extract plot options from kwargs
        for k in util.denone(kw.get("PlotOptions", {})):
            # Override the default option.
            if kw["PlotOptions"][k] is not None:
                kw_p[k] = kw["PlotOptions"][k]
        # Plot the coefficient
        h[c] = plt.plot(iters[j0:jB+1], C[j0:jB+1], **kw_p)
        # Get the figure and axes.
        h['fig'] = plt.gcf()
        h['ax'] = plt.gca()
        # Check for an existing ylabel
        ly = h['ax'].get_ylabel()
        # Compare to the requested ylabel
        if ly and ly != c:
            # Combine labels
            ly = ly + '/' + c
        else:
            # Use the coefficient
            ly = c
        # Process axis labels
        xlbl = kw.get('XLabel', 'Iteration Number')
        ylbl = kw.get('YLabel', ly)
        # Labels.
        h['x'] = plt.xlabel(xlbl)
        h['y'] = plt.ylabel(ylbl)
        # Set the xlimits.
        h['ax'].set_xlim((i0, 1.03*iB-0.03*i0))
        # Set figure dimensions
        if fh:
            h['fig'].set_figheight(fh)
        if fw:
            h['fig'].set_figwidth(fw)
       # ------
       # Labels
       # ------
        # y-coordinates of the current axes w.r.t. figure scale
        ya = h['ax'].get_position().get_points()
        ha = ya[1, 1] - ya[0, 1]
        # y-coordinates above and below the box
        yf = 2.5 / ha / h['fig'].get_figheight()
        yu = 1.0 + 0.065*yf
        yl = 1.0 - 0.04*yf
        # Process options for label
        qlmu  = kw.get("ShowMu", True)
        qldel = kw.get("ShowDelta", True)
        qlsig = kw.get("ShowSigma", True)
        qlerr = kw.get("ShowError", True)
        # Further processing
        qldel = (dc and qldel)
        qlsig = (nAvg > 2) and ((ksig and qlsig) or kw.get("ShowSigma", False))
        qlerr = (nAvg > 6) and ((uerr and qlerr) or kw.get("ShowError", False))
        # Make a label for the mean value
        if qlmu:
            # printf-style format flag
            flbl = kw.get("MuFormat", "%.4f")
            # Form: CA = 0.0204
            lbl = (u'%s = %s' % (c, flbl)) % cAvg
            # Create the handle
            h['mu'] = plt.text(
                0.99, yu, lbl,
                color=kw_p['color'],
                horizontalalignment='right',
                verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font
            _set_font(h['mu'])
        # Make a label for the deviation.
        if qldel:
            # printf-style flag
            flbl = kw.get("DeltaFormat", "%.4f")
            # Form: \DeltaCA = 0.0050
            lbl = (u'\u0394%s = %s' % (c, flbl)) % dc
            # Create the handle.
            h['d'] = plt.text(
                0.99, yl, lbl,
                color=kw_d.get_opt('color', 1),
                horizontalalignment='right',
                verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font
            _set_font(h['d'])
        # Make a label for the standard deviation.
        if qlsig:
            # Printf-style flag
            flbl = kw.get("SigmaFormat", "%.4f")
            # Form \sigma(CA) = 0.0032
            lbl = (u'\u03C3(%s) = %s' % (c, flbl)) % c_std
            # Create the handle.
            h['sig'] = plt.text(
                0.01, yu, lbl,
                color=kw_s.get_opt('color', 1),
                horizontalalignment='left',
                verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font
            _set_font(h['sig'])
        # Make a label for the iterative uncertainty.
        if qlerr:
            # printf-style format flag
            flbl = kw.get("ErrorFormat", "%.4f")
            # Form \varepsilon(CA) = 0.0032
            lbl = (u'u(%s) = %s' % (c, flbl)) % c_err
            # Check position
            if qlsig:
                # Put below the upper border
                yerr = yl
            else:
                # Put above the upper border if there's no sigma in the way
                yerr = yu
            # Create the handle.
            h['eps'] = plt.text(
                0.01, yerr, lbl,
                color=kw_u.get_opt('color', 1),
                horizontalalignment='left',
                verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font
            _set_font(h['eps'])
       # -----------
       # Grid Lines
       # -----------
        # Get grid option
        ogrid = kw.get("Grid")
        # Check value
        if ogrid is None:
            # Leave it as it currently is
            pass
        elif ogrid:
            # Get grid style
            kw_g = kw.get("GridStyle", {})
            # Ensure that the axis is below
            h['ax'].set_axisbelow(True)
            # Add the grid
            h['ax'].grid(**kw_g)
        else:
            # Turn the grid off, even if previously turned on
            h['ax'].grid(False)
        # Get grid option
        ogrid = kw.get("MinorGrid")
        # Check value
        if ogrid is None:
            # Leave it as it currently is
            pass
        elif ogrid:
            # Get grid style
            kw_g = kw.get("MinorGridStyle", {})
            # Ensure that the axis is below
            h['ax'].set_axisbelow(True)
            # Minor ticks are required
            h['ax'].minorticks_on()
            # Add the grid
            h['ax'].grid(which="minor", **kw_g)
        else:
            # Turn the grid off, even if previously turned on
            h['ax'].grid(False)
       # ------------------
       # Ticks/Tick Labels
       # ------------------
        # Get *Ticks* option
        tck = kw.get("Ticks")
        xtck = kw.get("XTicks", tck)
        ytck = kw.get("YTicks", tck)
        # Get *TickLabels* option
        TL = kw.get("TickLabels")
        xTL = kw.get("XTickLabels", TL)
        yTL = kw.get("YTickLabels", TL)
        # Process x-axis ticks
        if xTL is None:
            # Do nothing
            pass
        elif xTL:
            # Manual list of tick labels (unlikely to work)
            h['ax'].set_xticklabels(xTL)
        else:
            # Turn axis labels off
            h['ax'].set_xticklabels([])
        # Process y-axis ticks
        if yTL is None:
            # Do nothing
            pass
        elif yTL:
            # Manual list of tick labels (unlikely to work)
            h['ax'].set_yticklabels(yTL)
        else:
            # Turn axis labels off
            h['ax'].set_yticklabels([])
        # Process x-axis ticks
        if xtck is None:
            # Do nothing
            pass
        elif xtck:
            # Manual list of tick labels (unlikely to work)
            h['ax'].set_xticks(xtck)
        else:
            # Turn axis labels off
            h['ax'].set_xticks([])
        # Process y-axis ticks
        if ytck is None:
            # Do nothing
            pass
        elif ytck:
            # Manual list of tick labels (unlikely to work)
            h['ax'].set_yticks(ytck)
        else:
            # Turn axis labels off
            h['ax'].set_yticks([])
       # -----------------
       # Final Formatting
       # -----------------
        # Attempt to apply tight axes
        _tight_layout()
        # Output
        return h

    # Plot coefficient histogram
    def PlotValueHist(self, coeff: str, nAvg=100, nLast=None, **kw):
        r"""Plot a histogram of the iterative history of some value *c*

        :Call:
            >>> h = fm.PlotValueHist(comp, c, n=1000, nAvg=100, **kw)
        :Inputs:
            *fm*: :class:`cape.cfdx.databook.CaseData`
                Instance of the component force history class
            *comp*: :class:`str`
                Name of component to plot
            *c*: :class:`str`
                Name of coefficient to plot, e.g. ``'CA'``
            *nAvg*: :class:`int`
                Use the last *nAvg* iterations to compute an average
            *nBins*: {``20``} | :class:`int`
                Number of bins in histogram, also can be set in *HistOptions*
            *nLast*: :class:`int`
                Last iteration to use (defaults to last iteration available)
        :Keyword Arguments:
            *FigureWidth*: :class:`float`
                Figure width
            *FigureHeight*: :class:`float`
                Figure height
            *Label*: [ {*comp*} | :class:`str` ]
                Manually specified label
            *TargetValue*: :class:`float` | :class:`list`\ [:class:`float`]
                Target or list of target values
            *TargetLabel*: :class:`str` | :class:`list` (:class:`str`)
                Legend label(s) for target(s)
            *StDev*: [ {None} | :class:`float` ]
                Multiple of iterative history standard deviation to plot
            *HistOptions*: :class:`dict`
                Plot options for the primary histogram
            *StDevOptions*: :class:`dict`
                Dictionary of plot options for the standard deviation plot
            *DeltaOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for reference range plot
            *MeanOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for mean line
            *TargetOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for target value lines
            *OutlierSigma*: {``7.0``} | :class:`float`
                Standard deviation multiplier for determining outliers
            *ShowMu*: :class:`bool`
                Option to print value of mean
            *ShowSigma*: :class:`bool`
                Option to print value of standard deviation
            *ShowError*: :class:`bool`
                Option to print value of sampling error
            *ShowDelta*: :class:`bool`
                Option to print reference value
            *ShowTarget*: :class:`bool`
                Option to show target value
            *MuFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the mean value
            *DeltaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the reference value *d*
            *SigmaFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the iterative standard deviation
            *TargetFormat*: {``"%.4f"``} | :class:`str`
                Format for text label of the target value
            *XLabel*: :class:`str`
                Specified label for *x*-axis, default is ``Iteration Number``
            *YLabel*: :class:`str`
                Specified label for *y*-axis, default is *c*
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2015-02-15 ``@ddalle``: v1.0
            * 2015-03-06 ``@ddalle``: v1.1; add *nLast*
            * 2015-03-06 ``@ddalle``: v1.2; change class
            * 2024-01-10 ``@ddalle``: v1.3; DataKit updates
        """
        # -----------
        # Preparation
        # -----------
        # Make sure the plotting modules are present.
        ImportPyPlot()
        # Initialize dictionary of handles.
        h = {}
        # Figure dimensions
        fw = kw.get('FigureWidth', 6)
        fh = kw.get('FigureHeight', 4.5)
        # ---------
        # Last Iter
        # ---------
        # Iterations
        I = self.get_values("i")
        # Most likely last iteration
        iB = I[-1]
        # Check for an input last iter
        if nLast is not None:
            # Attempt to use requested iter.
            if nLast < iB:
                # Using an earlier iter; make sure to use one in the hist.
                # Find the iterations that are less than i.
                jB = self.GetIterationIndex(nLast)
                iB = I[jB]
        # Get the index of *iB* in *fm.i*.
        jB = self.GetIterationIndex(iB)
        # --------------
        # Averaging Iter
        # --------------
        # Get the first iteration to use in averaging.
        iA = max(0, iB - nAvg) + 1
        # Make sure *iV* is in *fm.i* and get the index.
        jA = self.GetIterationIndex(iA)
        # Reselect *iV* in case initial value was not in *fm.i*.
        iA = I[jA]
        # -----
        # Stats
        # -----
        # Calculate # of independent samples
        # Extract the values
        V = self.get_values(coeff)
        # Check
        if V is None:
            raise KeyError(f"Could not find coeff '{coeff}'")
        # Apply filter
        V = V[jA:jB+1]
        # Calculate basic statistics
        vmu = np.mean(V)
        vstd = np.std(V)
        # Check for outliers ...
        ostd = kw.get('OutlierSigma', 7.0)
        # Apply outlier tolerance
        if ostd:
            # Find indices of cases that are within outlier range
            J = np.abs(V-vmu)/vstd <= ostd
            # Downselect
            V = V[J]
            # Recompute statistics
            vmu = np.mean(V)
            vstd = np.std(V)
        # Uncertainty options
        ksig = kw.get('StDev')
        # Reference delta
        dc = kw.get('Delta', 0.0)
        # Target values and labels
        vtarg = kw.get('TargetValue')
        ltarg = kw.get('TargetLabel')
        # Convert target values to list
        if vtarg in (None, False):
            vtarg = []
        elif not isinstance(vtarg, (list, tuple, np.ndarray)):
            vtarg = [vtarg]
        # Create appropriate target list for
        if not isinstance(ltarg, (list, tuple, np.ndarray)):
            ltarg = [ltarg]
        # --------------
        # Histogram Plot
        # --------------
        # Initialize plot options for histogram.
        kw_h = DBPlotOpts(
            facecolor='c',
            zorder=2,
            bins=kw.get('nBins', 20))
        # Extract options from kwargs
        for k in util.denone(kw.get("HistOptions", {})):
            # Override the default option.
            if kw["HistOptions"][k] is not None:
                kw_h[k] = kw["HistOptions"][k]
        # Check for range based on standard deviation
        if kw.get("Range"):
            # Use this number of pair of numbers as multiples of *vstd*
            r = kw["Range"]
            # Check for single number or list
            if isinstance(r, (list, tuple, np.ndarray)):
                # Separate lower and upper limits
                vmin = vmu - r[0]*vstd
                vmax = vmu + r[1]*vstd
            else:
                # Use as a single number
                vmin = vmu - r*vstd
                vmax = vmu + r*vstd
            # Overwrite any range option in *kw_h*
            kw_h['range'] = (vmin, vmax)
        # Plot the historgram.
        h['hist'] = plt.hist(V, **kw_h)
        # Get the figure and axes.
        h['fig'] = plt.gcf()
        h['ax'] = plt.gca()
        # Get current axis limits
        pmin, pmax = h['ax'].get_ylim()
        # Determine whether or not the distribution is normed
        q_normed = kw_h.get("normed", True)
        # Determine whether or not the bars are vertical
        q_vert = kw_h.get("orientation", "vertical") == "vertical"
        # ---------
        # Mean Plot
        # ---------
        # Option whether or not to plot mean as vertical line.
        if kw.get("PlotMean", True):
            # Initialize options for mean plot
            kw_m = DBPlotOpts(color='k', lw=2, zorder=6)
            kw_m["label"] = "Mean value"
            # Extract options from kwargs
            for k in util.denone(kw.get("MeanOptions", {})):
                # Override the default option.
                if kw["MeanOptions"][k] is not None:
                    kw_m[k] = kw["MeanOptions"][k]
            # Check orientation
            if q_vert:
                # Plot a vertical line for the mean.
                h['mean'] = plt.plot([vmu, vmu], [pmin, pmax], **kw_m)
            else:
                # Plot a horizontal line for th emean.
                h['mean'] = plt.plot([pmin, pmax], [vmu, vmu], **kw_m)
        # -----------
        # Target Plot
        # -----------
        # Option whether or not to plot targets
        if vtarg is not None and len(vtarg) > 0:
            # Initialize options for target plot
            kw_t = DBPlotOpts(color='k', lw=2, ls='--', zorder=8)
            # Set label
            if ltarg is not None:
                # User-specified list of labels
                kw_t["label"] = ltarg
            else:
                # Default label
                kw_t["label"] = "Target"
            # Extract options for target plot
            for k in util.denone(kw.get("TargetOptions", {})):
                # Override the default option.
                if kw["TargetOptions"][k] is not None:
                    kw_t[k] = kw["TargetOptions"][k]
            # Loop through target values
            for i in range(len(vtarg)):
                # Select the value
                vt = vtarg[i]
                # Check for NaN or None
                if np.isnan(vt) or vt in (None, False):
                    continue
                # Downselect options
                kw_ti = {}
                for k in kw_t:
                    kw_ti[k] = kw_t.get_opt(k, i)
                # Initialize handles
                h['target'] = []
                # Check orientation
                if q_vert:
                    # Plot a vertical line for the target.
                    h['target'].append(
                        plt.plot([vt, vt], [pmin, pmax], **kw_ti))
                else:
                    # Plot a horizontal line for the target.
                    h['target'].append(
                        plt.plot([pmin, pmax], [vt, vt], **kw_ti))
        # -----------------------
        # Standard Deviation Plot
        # -----------------------
        # Check whether or not to plot it
        if ksig and len(I) > 2:
            # Check for single number or list
            if isinstance(ksig, (np.ndarray, list, tuple)):
                # Separate lower and upper limits
                vmin = vmu - ksig[0]*vstd
                vmax = vmu + ksig[1]*vstd
            else:
                # Use as a single number
                vmin = vmu - ksig*vstd
                vmax = vmu + ksig*vstd
            # Initialize options for std plot
            kw_s = DBPlotOpts(color='b', lw=2, zorder=5)
            # Extract options from kwargs
            for k in util.denone(kw.get("StDevOptions", {})):
                # Override the default option.
                if kw["StDevOptions"][k] is not None:
                    kw_s[k] = kw["StDevOptions"][k]
            # Check orientation
            if q_vert:
                # Plot a vertical line for the min and max
                h['std'] = (
                    plt.plot([vmin, vmin], [pmin, pmax], **kw_s) +
                    plt.plot([vmax, vmax], [pmin, pmax], **kw_s))
            else:
                # Plot a horizontal line for the min and max
                h['std'] = (
                    plt.plot([pmin, pmax], [vmin, vmin], **kw_s) +
                    plt.plot([pmin, pmax], [vmax, vmax], **kw_s))
        # ----------
        # Delta Plot
        # ----------
        # Check whether or not to plot it
        if dc:
            # Initialize options for delta plot
            kw_d = DBPlotOpts(color="r", ls="--", lw=1.0, zorder=3)
            # Extract options from kwargs
            for k in util.denone(kw.get("DeltaOptions", {})):
                # Override the default option.
                if kw["DeltaOptions"][k] is not None:
                    kw_d[k] = kw["DeltaOptions"][k]
                # Check for single number or list
            if type(dc).__name__ in ['ndarray', 'list', 'tuple']:
                # Separate lower and upper limits
                cmin = vmu - dc[0]
                cmax = vmu + dc[1]
            else:
                # Use as a single number
                cmin = vmu - dc
                cmax = vmu + dc
            # Check orientation
            if q_vert:
                # Plot vertical lines for the reference length
                h['delta'] = (
                    plt.plot([cmin, cmin], [pmin, pmax], **kw_d) +
                    plt.plot([cmax, cmax], [pmin, pmax], **kw_d))
            else:
                # Plot horizontal lines for reference length
                h['delta'] = (
                    plt.plot([pmin, pmax], [cmin, cmin], **kw_d) +
                    plt.plot([pmin, pmax], [cmax, cmax], **kw_d))
        # ----------
        # Formatting
        # ----------
        # Default value-axis label
        lx = coeff
        # Default probability-axis label
        if q_normed:
            # Size of bars is probability
            ly = "Probability Density"
        else:
            # Size of bars is count
            ly = "Count"
        # Process axis labels
        xlbl = kw.get('XLabel')
        ylbl = kw.get('YLabel')
        # Apply defaults
        if xlbl is None:
            xlbl = lx
        if ylbl is None:
            ylbl = ly
        # Check for flipping
        if not q_vert:
            xlbl, ylbl = ylbl, xlbl
        # Labels.
        h['x'] = plt.xlabel(xlbl)
        h['y'] = plt.ylabel(ylbl)
        # Set figure dimensions
        if fh:
            h['fig'].set_figheight(fh)
        if fw:
            h['fig'].set_figwidth(fw)
        # Attempt to apply tight axes
        _tight_layout()
        # ------
        # Labels
        # ------
        # y-coordinates of the current axes w.r.t. figure scale
        ya = h['ax'].get_position().get_points()
        ha = ya[1, 1] - ya[0, 1]
        # y-coordinates above and below the box
        yf = 2.5 / ha / h['fig'].get_figheight()
        yu = 1.0 + 0.065*yf
        yl = 1.0 - 0.04*yf
        # Make a label for the mean value.
        if kw.get("ShowMu", True):
            # printf-style format flag
            flbl = kw.get("MuFormat", "%.4f")
            # Form: CA = 0.0204
            lbl = (u'%s = %s' % (coeff, flbl)) % vmu
            # Create the handle.
            h['mu'] = plt.text(
                0.99, yu, lbl,
                color=kw_m['color'],
                horizontalalignment='right',
                verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font
            _set_font(h['mu'])
        # Make a label for the deviation.
        if dc and kw.get("ShowDelta", True):
            # printf-style flag
            flbl = kw.get("DeltaFormat", "%.4f")
            # Form: \DeltaCA = 0.0050
            lbl = (u'\u0394%s = %s' % (coeff, flbl)) % dc
            # Create the handle.
            h['d'] = plt.text(
                0.01, yl, lbl,
                color=kw_d.get_opt('color', 1),
                horizontalalignment='left',
                verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font
            _set_font(h['d'])
        # Make a label for the standard deviation.
        if len(I) > 2 and (
                (ksig and kw.get("ShowSigma", True)) or
                kw.get("ShowSigma", False)):
            # Printf-style flag
            flbl = kw.get("SigmaFormat", "%.4f")
            # Form \sigma(CA) = 0.0032
            lbl = (u'\u03C3(%s) = %s' % (coeff, flbl)) % vstd
            # Create the handle.
            h['sig'] = plt.text(
                0.01, yu, lbl,
                color=kw_s.get_opt('color', 1),
                horizontalalignment='left',
                verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font
            _set_font(h['sig'])
        # Make a label for the iterative uncertainty.
        if len(vtarg) > 0 and kw.get("ShowTarget", True):
            # printf-style format flag
            flbl = kw.get("TargetFormat", "%.4f")
            # Form Target = 0.0032
            lbl = (u'%s = %s' % (ltarg[0], flbl)) % vtarg[0]
            # Create the handle.
            h['t'] = plt.text(
                0.99, yl, lbl,
                color=kw_t.get_opt('color', 0),
                horizontalalignment='right',
                verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font
            _set_font(h['t'])
        # Output.
        return h


# Individual component force and moment
class CaseFM(CaseData):
    r"""Force and moment iterative histories

    This class contains methods for reading data about an the histroy of
    an individual component for a single case.

    :Call:
        >>> fm = cape.cfdx.databook.CaseFM(C, MRP=None, A=None)
    :Inputs:
        *C*: :class:`list` (:class:`str`)
            List of coefficients to initialize
        *MRP*: :class:`numpy.ndarray`\ [:class:`float`] shape=(3,)
            Moment reference point
        *A*: :class:`numpy.ndarray` shape=(*N*,4) or shape=(*N*,7)
            Matrix of forces and/or moments at *N* iterations
    :Outputs:
        *fm*: :class:`cape.aero.FM`
            Instance of the force and moment class
        *fm.coeffs*: :class:`list` (:class:`str`)
            List of coefficients
        *fm.MRP*: :class:`numpy.ndarray`\ [:class:`float`] shape=(3,)
            Moment reference point
    :Versions:
        * 2014-11-12 ``@ddalle``: Starter version
        * 2014-12-21 ``@ddalle``: Copied from previous `aero.FM`
    """
   # --- Class attributes ---
    # Attributes
    __slots__ = (
        "comp",
    )

    # Minimal list of columns
    _base_cols = (
        "i",
        "solver_iter",
        "CA",
        "CY",
        "CN",
        "CLL",
        "CLM",
        "CLN",
    )
    # Minimal list of "coeffs"
    _base_coeffs = (
        "CA",
        "CY",
        "CN",
        "CLL",
        "CLM",
        "CLN",
    )

   # --- __dunder__ ---
    # Initialization method
    def __init__(self, comp: str, **kw):
        r"""Initialization method

        :Versions:
            * 2014-11-12 ``@ddalle``: v1.0
            * 2015-10-16 ``@ddalle``: v1.1; trivial generic version
            * 2024-01-22 ``@ddalle``: v2.0; DataKit + caching
        """
        # Save the component name
        self.comp = comp
        # Call parent initialization
        CaseData.__init__(self, **kw)

    # Function to display contents
    def __repr__(self):
        r"""Representation method

        Returns the following format, with ``'entire'`` replaced with the
        component name, *fm.comp*

            * ``'<CaseFM('entire', i=100)>'``

        :Versions:
            * 2014-11-12 ``@ddalle``: v1.0
            * 2015-10-16 ``@ddalle``: v2.0; generic version
            * 2024-01-21 ``@ddalle``: v2.1; even more generic
        """
        return "<%s('%s', i=%i)>" % (
            self.__class__.__name__, self.comp, self["i"].size)
    # String method
    __str__ = __repr__

   # --- I/O ---
    # Get cache file name
    def get_cdbfile(self) -> str:
        r"""Get path to iterative history cache file

        :Call:
            >>> cdbfilename = h.get_cdbfile()
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history instance
        :Outputs:
            *cdbfilename*: :class:`str`
                Name of file (extension is ``.cdb``)
        :Versions:
            * 2024-01-22 ``@ddalle``: v1.0
        """
        # Get file name
        return os.path.join("cape", f"fm_{self.comp}.cdb")

   # --- Data ---
    # Copy
    def Copy(self):
        r"""Copy an iterative force & moment history

        :Call:
            >>> fm2 = FM1.Copy()
        :Inputs:
            *FM1*: :class:`cape.cfdx.databook.CaseFM`
                Force and moment history
        :Outputs:
            *FM2*: :class:`cape.cfdx.databook.CaseFM`
                Copy of *FM1*
        :Versions:
            * 2017-03-20 ``@ddalle``: v1.0
            * 2024-01-10 ``@ddalle``: v2.0; simplify using DataKit
        """
        # Initialize output
        fm = CaseFM(self.comp)
        # Link
        fm.link_data(self)
        # Output
        return fm

    # Method to add data to instance
    def AddData(self, A: dict):
        r"""Add iterative force and/or moment history for a component

        :Call:
            >>> fm.AddData(A)
        :Inputs:
            *fm*: :class:`cape.cfdx.databook.CaseFM`
                Instance of the force and moment class
            *A*: :class:`numpy.ndarray` shape=(*N*,4) or shape=(*N*,7)
                Matrix of forces and/or moments at *N* iterations
        :Versions:
            * 2014-11-12 ``@ddalle``: v1.0
            * 2015-10-16 ``@ddalle``: v2.0; complete rewrite
            * 2024-01-10 ``@ddalle``: v2.1; simplify using DataKit
        """
        self.link_data(A)

   # ============
   # Operations
   # ============
   # <
    # Trim repeated iterations
    def trim_iters(self):
        r"""Trim any repeated iterations from history

        :Call:
            >>> db.trim_iters()
        :Inputs:
            *db*; :class:`cape.cfdx.casedata.CaseData`
                Single-case iterative history
        :Versions:
            * 2025-07-31 ``@ddalle``: v1.0
        """
        # Get iteration column
        col = CASE_COL_ITERS
        # Find unique ascending values (keeping last copy of duplicates)
        mask = self.find_ascending(col, keep_last=True)
        # Apply it
        self.apply_mask_iters(mask)

    # Trim entries
    def TrimIters(self):
        r"""Trim non-ascending iterations and other problems

        :Call:
            >>> fm.TrimIters()
        :Versions:
            * 2017-10-02 ``@ddalle``: v1.0
            * 2024-01-10 ``@ddalle``: v2.0; DataKit updates
        """
        # Get iterations
        iters = self.get_values("i")
        # Do nothing if not present
        if iters is None:
            return
        # Number of existing iterations
        n = len(iters)
        # Initialize iterations to keep
        mask = np.ones(n, dtype="bool")
        # Last iteration available
        i1 = iters[-1]
        # Loop through iterations
        for j in range(n-1):
            # Check for any following iterations that are less than this
            mask[j] = (iters[j] <= np.min(iters[j+1:]))
            # Check for last iteration less than current
            mask[j] = (mask[j] and iters[j] < i1)
        # Perform trimming actions
        for col in self.cols:
            self[col] = self.get_values(col, mask)

    # Add components
    def __add__(self, fm: CaseData):
        r"""Add two iterative histories

        :Call:
            >>> fm3 = fm1.__add__(fm2)
            >>> fm3 = fm1 + fm2
        :Inputs:
            *fm1*: :class:`cape.cfdx.databook.CaseFM`
                Initial force and moment iterative history
            *fm2*: :class:`cape.cfdx.databook.CaseFM`
                Second force and moment iterative history
        :Outputs:
            *fm1*: :class:`cape.cfdx.databook.CaseFM`
                Iterative history attributes other than iter numbers are added
        :Versions:
            * 2017-03-20 ``@ddalle``: v1.0
            * 2024-01-10 ``@ddalle``: v1.1; DataKit updates
        """
        # Get iterations list
        selfi = self.get_values("i")
        fmi = fm.get_values("i")
        # Check dimensions
        if (selfi.size != fmi.size) or np.any(selfi != fmi):
            # Trim any reversions of iterations
            self.TrimIters()
            fm.TrimIters()
        # Check dimensions
        if selfi.size > fmi.size:
            raise IndexError(
                ("Cannot add iterative F&M histories\n  %s\n" % self) +
                ("  %s\ndue to inconsistent size" % fm))
        # Create a copy
        fm3 = self.Copy()
        # Loop through columns
        for col in self.coeffs:
            # Number of values in this object
            n = len(self[col])
            # Get value
            v = fm[col]
            # Check type
            if not isinstance(v, np.ndarray):
                continue
            # Update the field
            fm3[col] = self[col] + v[:n]
        # Output
        return fm3

    # Add in place
    def __iadd__(self, fm: CaseData):
        r"""Add a second iterative history in place

        :Call:
            >>> fm1 = fm1.__iadd__(fm2)
            >>> fm1 += fm2
        :Inputs:
            *fm1*: :class:`cape.cfdx.databook.CaseFM`
                Initial force and moment iterative history
            *fm2*: :class:`cape.cfdx.databook.CaseFM`
                Second force and moment iterative history
        :Outputs:
            *fm1*: :class:`cape.cfdx.databook.CaseFM`
                Iterative history attributes other than iter numbers are added
        :Versions:
            * 2017-03-20 ``@ddalle``: v1.0
            * 2024-01-10 ``@ddalle``: v1.1; DataKit updates
        """
        # Get iterations
        selfi = self.get_values("i")
        fmi = self.get_values("i")
        # Check dimensions
        if (selfi.size != fmi.size) or np.any(selfi != fmi):
            # Trim any reversions of iterations
            self.TrimIters()
            fm.TrimIters()
        # Check dimensions
        if selfi.size > fmi.size:
            # Trim all
            for col in ("i", "CA", "CY", "CN", "CLL", "CLM", "CLN"):
                self[col] = self[col][:fmi.size]
        # Loop through columns
        for col in self.coeffs:
            # Get value
            v = fm[col]
            # Number of values in this object
            na = len(self[col])
            nb = len(v)
            n = min(na, nb)
            # Check type
            if not isinstance(v, np.ndarray):
                continue
            # Update the field
            self[col] = self[col][:n] + v[:n]
        # Apparently you need to output
        return self

    # Subtract components
    def __sub__(self, fm: CaseData):
        r"""Add two iterative histories

        :Call:
            >>> fm3 = FM1.__sub__(FM2)
            >>> fm3 = FM1 - FM2
        :Inputs:
            *FM1*: :class:`cape.cfdx.databook.CaseFM`
                Initial force and moment iterative history
            *FM2*: :class:`cape.cfdx.databook.CaseFM`
                Second force and moment iterative history
        :Outputs:
            *FM1*: :class:`cape.cfdx.databook.CaseFM`
                Iterative history attributes other than iter numbers are added
        :Versions:
            * 2017-03-20 ``@ddalle``: v1.0
            * 2024-01-10 ``@ddalle``: v1.1; DataKit updates
        """
        # Get iterations
        selfi = self.get_values("i")
        fmi = self.get_values("i")
        # Check dimensions
        if (selfi.size != fmi.size) or np.any(selfi != fmi):
            # Trim any reversions of iterations
            self.TrimIters()
            fm.TrimIters()
        # Check dimensions
        if selfi.size > fmi.size:
            raise IndexError(
                ("Cannot subtract iterative F&M histories\n  %s\n" % self) +
                ("  %s\ndue to inconsistent size" % fm))
        # Create a copy
        fm3 = self.Copy()
        # Loop through columns
        for col in self.coeffs:
            # Number of values in this object
            n = len(self[col])
            # Get value
            v = fm[col]
            # Check type
            if not isinstance(v, np.ndarray):
                continue
            # Update the field
            fm3[col] = self[col] - v[:n]
        # Output
        return fm3

    # Add in place
    def __isub__(self, fm: CaseData):
        r"""Add a second iterative history in place

        :Call:
            >>> fm1 = fm1.__isub__(fm2)
            >>> fm1 -= fm2
        :Inputs:
            *FM1*: :class:`cape.cfdx.databook.CaseFM`
                Initial force and moment iterative history
            *FM2*: :class:`cape.cfdx.databook.CaseFM`
                Second force and moment iterative history
        :Outputs:
            *FM1*: :class:`cape.cfdx.databook.CaseFM`
                Iterative history attributes other than iter numbers are added
        :Versions:
            * 2017-03-20 ``@ddalle``: v1.0
            * 2024-01-10 ``@ddalle``: v1.1; DataKit updates
        """
        # Get iterations
        selfi = self.get_values("i")
        fmi = self.get_values("i")
        # Check dimensions
        if (selfi.size != fmi.size) or np.any(selfi != fmi):
            # Trim any reversions of iterations
            self.TrimIters()
            fm.TrimIters()
        # Check dimensions
        if selfi.size > fmi.size:
            raise IndexError(
                ("Cannot subtract iterative F&M histories\n  %s\n" % self) +
                ("  %s\ndue to inconsistent size" % fm))
        # Loop through columns
        for col in self.coeffs:
            # Number of values in this object
            n = len(self[col])
            # Get value
            v = fm[col]
            # Check type
            if not isinstance(v, np.ndarray):
                continue
            # Update the field
            self[col] -= fm[col][:n]
        # Apparently you need to output
        return self
   # >

   # =================
   # Transformations
   # =================
   # <
    # Transform force or moment reference frame
    def TransformFM(self, topts: dict, x: dict, i: int):
        r"""Transform a force and moment history

        Available transformations and their parameters are listed below.

            * "Euler321": "psi", "theta", "phi"
            * "Euler123": "phi", "theta", "psi"
            * "ScaleCoeffs": "CA", "CY", "CN", "CLL", "CLM", "CLN"

        RunMatrix variables are used to specify values to use for the
        transformation variables.  For example,

            .. code-block:: python

                topts = {"Type": "Euler321",
                    "psi": "Psi", "theta": "Theta", "phi": "Phi"}

        will cause this function to perform a reverse Euler 3-2-1
        transformation using *x.Psi[i]*, *x.Theta[i]*, and *x.Phi[i]* as
        the angles.

        Coefficient scaling can be used to fix incorrect reference areas
        or flip axes. The default is actually to flip *CLL* and *CLN*
        due to the transformation from CFD axes to standard flight
        dynamics axes.

            .. code-block:: python

                tops = {"Type": "ScaleCoeffs",
                    "CLL": -1.0, "CLN": -1.0}

        :Call:
            >>> fm.TransformFM(topts, x, i)
        :Inputs:
            *fm*: :class:`cape.cfdx.databook.CaseFM`
                Instance of the force and moment class
            *topts*: :class:`dict`
                Dictionary of options for the transformation
            *x*: :class:`cape.runmatrix.RunMatrix`
                The run matrix used for this analysis
            *i*: :class:`int`
                Run matrix case index
        :Versions:
            * 2014-12-22 ``@ddalle``: v1.0
        """
        # Get the transformation type.
        ttype = topts.get("Type", "")
        # Check it.
        if ttype in ["Euler321", "Euler123"]:
            # Get the angle variable names
            kph = topts.get('phi', 0.0)
            kth = topts.get('theta', 0.0)
            kps = topts.get('psi', 0.0)
            # Extract roll
            if not isinstance(kph, str):
                # Fixed value
                phi = kph*deg
            elif kph.startswith('-'):
                # Negative roll angle.
                phi = -x[kph[1:]][i]*deg
            else:
                # Positive roll
                phi = x[kph][i]*deg
            # Extract pitch
            if not isinstance(kth, str):
                # Fixed value
                theta = kth*deg
            elif kth.startswith('-'):
                # Negative pitch
                theta = -x[kth[1:]][i]*deg
            else:
                # Positive pitch
                theta = x[kth][i]*deg
            # Extract yaw
            if not isinstance(kps, str):
                # Fixed value
                psi = kps*deg
            elif kps.startswith('-'):
                # Negative yaw
                psi = -x[kps[1:]][i]*deg
            else:
                # Positive pitch
                psi = x[kps][i]*deg
            # Sines and cosines
            cph, cth, cps = np.cos(phi), np.cos(theta), np.cos(psi)
            sph, sth, sps = np.sin(phi), np.sin(theta), np.sin(psi)
            # Make the matrices.
            # Roll matrix
            R1 = np.array([[1, 0, 0], [0, cph, -sph], [0, sph, cph]])
            # Pitch matrix
            R2 = np.array([[cth, 0, -sth], [0, 1, 0], [sth, 0, cth]])
            # Yaw matrix
            R3 = np.array([[cps, -sps, 0], [sps, cps, 0], [0, 0, 1]])
            # Combined transformation matrix.
            # Remember, these are applied backwards in order to undo the
            # original Euler transformation that got the component here.
            if ttype == "Euler321":
                R = np.dot(R1, np.dot(R2, R3))
            elif ttype == "Euler123":
                R = np.dot(R3, np.dot(R2, R1))
            # Force transformations
            if 'CY' in self.coeffs:
                # Assemble forces
                CA = self.get_values("CA")
                CY = self.get_values("CY")
                CN = self.get_values("CN")
                Fc = np.vstack((CA, CY, CN))
                # Transform.
                Fb = np.dot(R, Fc)
                # Extract (is this necessary?)
                self["CA"] = Fb[0]
                self["CY"] = Fb[1]
                self["CN"] = Fb[2]
            elif 'CN' in self.coeffs:
                # Use zeros for side force
                CA = self.get_values("CA")
                CN = self.get_values("CN")
                CY = np.zeros_like(CN)
                # Assemble forces.
                Fc = np.vstack((CA, CY, CN))
                # Transform.
                Fb = np.dot(R, Fc)
                # Extract
                self["CA"] = Fb[0]
                self["CN"] = Fb[2]
            # Moment transformations
            if 'CLN' in self.coeffs:
                # Get moments
                CLL = self.get_values("CLL")
                CLM = self.get_values("CLM")
                CLN = self.get_values("CLN")
                # Assemble moment vector
                Mc = np.vstack((CLL, CLM, CLN))
                # Transform.
                Mb = np.dot(R, Mc)
                # Extract.
                self["CLL"] = Mb[0]
                self["CLM"] = Mb[1]
                self["CLN"] = Mb[2]
            elif 'CLM' in self.coeffs:
                # Use zeros for roll and yaw moment
                CLM = self.get_values("CLM")
                CLL = np.zeros_like(CLM)
                CLN = np.zeros_like(CLN)
                # Assemble moment vector.
                Mc = np.vstack((CLL, CLM, CLN))
                # Transform.
                Mb = np.dot(R, Mc)
                # Extract.
                self["CLM"] = Mb[1]
        elif ttype in ["ScaleCoeffs"]:
            # Loop through coefficients.
            for c in topts:
                # Check if it's an available coefficient.
                if c not in self.coeffs:
                    continue
                # Get the value
                k = topts[c]
                # Check if it's a number
                kcls = type(k).__name__
                if not (kcls.startswith("float") or kcls.startswith("int")):
                    # Assume they meant to flip it.
                    k = -1.0
                # Scale.
                self[c] *= k
        elif ttype in ["ShiftMRP"]:
            # Get target MRP
            x0 = topts.get("FromMRP")
            x1 = topts.get("ToMRP")
            Lref = topts.get("RefLength")
            # Transform
            self.ShiftMRP(Lref, x1, x0)
        else:
            raise IOError(
                "Transformation type '%s' is not recognized." % ttype)

    # Method to shift the MRC
    def ShiftMRP(self, Lref, x, xi=None):
        r"""Shift the moment reference point

        :Call:
            >>> fm.ShiftMRP(Lref, x, xi=None)
        :Inputs:
            *fm*: :class:`cape.cfdx.databook.CaseFM`
                Instance of the force and moment class
            *Lref*: :class:`float`
                Reference length
            *x*: :class:`list`\ [:class:`float`]
                Target moment reference point
            *xi*: :class:`list`\ [:class:`float`]
                Current moment reference point (default: *self.MRP*)
        :Versions:
            * 2015-03-02 ``@ddalle``: v1.0
        """
        # Check for moments.
        if ('CA' not in self.coeffs) or ('CLM' not in self.coeffs):
            # Not a force/moment history
            return
        # Rolling moment: side force
        if ('CLL' in self.coeffs) and ('CY' in self.coeffs):
            self["CLL"] -= (xi[2]-x[2])/Lref*self["CY"]
        # Rolling moment: normal force
        if ('CLL' in self.coeffs) and ('CN' in self.coeffs):
            self["CLL"] += (xi[1]-x[1])/Lref*self["CN"]
        # Pitching moment: normal force
        if ('CLM' in self.coeffs) and ('CN' in self.coeffs):
            self["CLM"] -= (xi[0]-x[0])/Lref*self["CN"]
        # Pitching moment: axial force
        if ('CLM' in self.coeffs) and ('CA' in self.coeffs):
            self["CLM"] += (xi[2]-x[2])/Lref*self["CA"]
        # Yawing moment: axial force
        if ('CLN' in self.coeffs) and ('CA' in self.coeffs):
            self["CLN"] -= (xi[1]-x[1])/Lref*self["CA"]
        # Yawing moment: axial force
        if ('CLN' in self.coeffs) and ('CY' in self.coeffs):
            self["CLN"] += (xi[0]-x[0])/Lref*self["CY"]

    # Shift the moment center for a history of points
    def shift_mrp_array(self, dat: dict):
        ...
   # >

   # ===========
   # Statistics
   # ===========
   # <
    # Method to get averages and standard deviations
    def GetStatsN(self, nStats=100, nLast=None):
        r"""Get mean, min, max, and standard deviation for all coefficients

        :Call:
            >>> s = fm.GetStatsN(nStats, nLast=None)
        :Inputs:
            *fm*: :class:`cape.cfdx.databook.CaseFM`
                Instance of the force and moment class
            *nStats*: :class:`int`
                Number of iterations in window to use for statistics
            *nLast*: :class:`int`
                Last iteration to use for statistics
        :Outputs:
            *s*: :class:`dict`\ [:class:`float`]
                Dictionary of mean, min, max, std for each coefficient
        :Versions:
            * 2014-12-09 ``@ddalle``: v1.0
            * 2015-02-28 ``@ddalle``: v1.1; was ``GetStats()``
            * 2015-03-04 ``@ddalle``: v1.2; add *nLast*
            * 2024-01-10 ``@ddalle``: v1.3; DataKit updates
        """
        # Get iterations
        iters = self.get_values("i")
        # Last iteration to use.
        if nLast:
            # Attempt to use requested iter.
            if iters.size == 0:
                # No iterations
                iLast = 0
            elif nLast < iters[-1]:
                # Using an earlier iter; make sure to use one in the hist.
                jLast = self.GetIterationIndex(nLast)
                # Find the iterations that are less than i.
                iLast = iters[jLast]
            else:
                # Use the last iteration.
                iLast = iters[-1]
        else:
            # Just use the last iteration
            iLast = iters[-1]
        # Get index
        jLast = self.GetIterationIndex(iLast)
        # Default values.
        if (nStats is None) or (nStats < 2):
            # Use last iteration
            i0 = iLast
        else:
            # Process min indices for plotting and averaging.
            i0 = max(0, iLast-nStats)
        # Get index
        j0 = self.GetIterationIndex(i0)
        # Initialize output.
        s = {}
        # Loop through coefficients.
        for c in self.coeffs:
            # Get the values
            F = self.get_values(c)
            # Save the mean value
            s[c] = np.mean(F[j0:jLast+1])
            # Check for statistics.
            if (nStats is not None) or (nStats < 2):
                # Save the statistics.
                if jLast <= j0:
                    # Print a nice error message
                    raise ValueError(
                        ("FM component '%s' has no iterations " % self.comp) +
                        ("for coefficient '%s'\n" % c) +
                        ("DataBook component '%s' has the " % self.comp) +
                        ("wrong type or is not being reported by the solver"))
                s[c+'_min'] = np.min(F[j0:jLast+1])
                s[c+'_max'] = np.max(F[j0:jLast+1])
                s[c+'_std'] = np.std(F[j0:jLast+1])
                s[c+'_err'] = util.SigmaMean(F[j0:jLast+1])
        # Output
        return s

    # Method to get averages and standard deviations
    def GetStatsOld(self, nStats=100, nMax=None, nLast=None):
        r"""Get mean, min, max, and standard deviation for all coefficients

        :Call:
            >>> s = fm.GetStatsOld(nStats, nMax=None, nLast=None)
        :Inputs:
            *fm*: :class:`cape.cfdx.databook.CaseFM`
                Instance of the force and moment class
            *nStats*: :class:`int`
                Minimum number of iterations in window to use for statistics
            *nMax*: :class:`int`
                Maximum number of iterations to use for statistics
            *nLast*: :class:`int`
                Last iteration to use for statistics
        :Outputs:
            *s*: :class:`dict`\ [:class:`float`]
                Dictionary of mean, min, max, std for each coefficient
        :Versions:
            * 2015-02-28 ``@ddalle``: v1.0
            * 2015-03-04 ``@ddalle``: v1.1; add *nLast*
            * 2024-01-10 ``@ddalle``: v1.2; DataKit updates
        """
        # Make sure the number of iterations used is an integer.
        if not nStats:
            nStats = 1
        # Process list of candidate numbers of iterations for statistics.
        if nMax and (nStats > 1) and (nMax >= 1.5*nStats):
            # Nontrivial list of candidates
            # Multiples of *nStats*
            N = [k*nStats for k in range(1, int(nMax/nStats)+1)]
            # Check if *nMax* should also be considered.
            if nMax >= 1.5*N[-1]:
                # Add *nMax*
                N.append(nMax)
        else:
            # Only one candidate.
            N = [nStats]
        # Initialize error as infinity.
        e = np.inf
        # Loop through list of candidate iteration counts
        for n in N:
            # Get the statistics.
            sn = self.GetStatsN(n, nLast=nLast)
            # Save the number of iterations used.
            sn['nStats'] = n
            # If there is only one candidate, return it.
            if len(N) == 1:
                return sn
            # Calculate the composite error.
            en = np.sqrt(np.sum([sn[c+'_err']**2 for c in self.coeffs]))
            # Calibrate to slightly favor less iterations
            en = en * (0.75 + 0.25*np.sqrt(n)/np.sqrt(N[0]))
            # Check if this error is an improvement.
            if (n == min(N)) or (en < e):
                # Select these statistics, and update the best scaled error.
                s = sn
                e = en
        # Output.
        return s

    # Get status for one coefficient
    def GetStatsCoeff(self, coeff, nStats=100, nMax=None, **kw):
        r"""Get mean, min, max, and other statistics for one coefficient

        :Call:
            >>> s = fm.GetStatsCoeff(coeff, nStats=100, nMax=None, **kw)
        :Inputs:
            *fm*: :class:`cape.cfdx.databook.CaseFM`
                Instance of the force and moment class
            *coeff*: :class:`str`
                Name of coefficient to process
            *nStats*: {``100``} | :class:`int`
                Min number of iterations in window to use for statistics
            *dnStats*: {*nStats*} | :class:`int`
                Interval size for candidate windows
            *nMax*: (*nStats*} | :class:`int`
                Maximum number of iterations to use for statistics
            *nMin*: {``0``} | :class:`int`
                First usable iteration number
            *nLast*: {*fm.i[-1]*} | :class:`int`
                Last iteration to use for statistics
        :Outputs:
            *s*: :class:`dict`\ [:class:`float`]
                Dictionary of mean, min, max, std for *coeff*
        :Versions:
            * 2017-09-29 ``@ddalle``: v1.0
            * 2024-01-10 ``@ddalle``: v1.1; DataKit updates
        """
        # Iterations
        iters = self.get_values(CASE_COL_ITERS)
        # Number of iterations available
        ni = len(iters)
        # Default last iteration
        if ni == 0:
            # No iterations
            nLast = 0
        else:
            # Last iteration
            nLast = iters[-1]
        # Read iteration values
        nLast = kw.get('nLast', nLast)
        # Get maximum size
        if nMax is None:
            nMax = nStats
        # Get interval size
        dnStats = kw.get("dnStats", nStats)
        # First usable iteration
        nMin = kw.get("nMin", 0)
        # Get coefficient
        F = self.ExtractValue(coeff, **kw)
        # Get statistics
        d = util.SearchSinusoidFitRange(
            iters, F, nStats, nMax,
            dn=dnStats, nMin=nMin)
        # Output
        return d

    # Method to get averages and standard deviations
    def GetStats(self, nStats=100, nMax=None, **kw):
        r"""Get mean, min, max, and stdev for all coefficients

        :Call:
            >>> s = fm.GetStats(nStats, nMax=None, nLast=None)
        :Inputs:
            *fm*: :class:`cape.cfdx.databook.CaseFM`
                Instance of the force and moment class
            *coeff*: :class:`str`
                Name of coefficient to process
            *nStats*: {``100``} | :class:`int`
                Min number of iterations in window to use for statistics
            *dnStats*: {*nStats*} | :class:`int`
                Interval size for candidate windows
            *nMax*: (*nStats*} | :class:`int`
                Maximum number of iterations to use for statistics
            *nMin*: {``0``} | :class:`int`
                First usable iteration number
            *nLast*: {*fm.i[-1]*} | :class:`int`
                Last iteration to use for statistics
        :Outputs:
            *s*: :class:`dict`\ [:class:`float`]
                Dictionary of mean, min, max, std, err for each
        :Versions:
            * 2017-09-29 ``@ddalle``: v1.0
            * 2024-01-10 ``@ddalle``: v1.1; DataKit updates
        """
        # Get iterations
        iters = self.get_values("i")
        # Check for empty instance
        if iters.size == 0:
            raise ValueError("No history found for comp '%s'\n" % self.comp)
        # Initialize output
        s = {}
        # Initialize statistics count
        ns = 0
        # Loop through coefficients
        for c in self.coeffs:
            # Get value
            v = self[c]
            # Check type
            if not isinstance(v, np.ndarray) or v.size == 0:
                continue
            # Get individual statistics
            d = self.GetStatsCoeff(c, nStats=nStats, nMax=nMax, **kw)
            # Transfer the information
            s[c] = d["mu"]
            s[c+'_n'] = d["n"]
            s[c+'_min'] = d["min"]
            s[c+'_max'] = d["max"]
            s[c+'_std'] = d["sig"]
            s[c+'_err'] = d["u"]
            # Update stats count
            ns = max(ns, d["n"])
        # Set the stats count
        s["nStats"] = ns
        # Output
        return s
   # >

   # ==========
   # Plotting
   # ==========
   # <
    # Plot iterative force/moment history
    def PlotCoeff(self, c: str, n=None, **kw):
        r"""Plot a single coefficient history

        :Call:
            >>> h = fm.PlotCoeff(c, n=1000, nAvg=100, **kw)
        :Inputs:
            *fm*: :class:`cape.cfdx.databook.CaseFM`
                Instance of the component force history class
            *c*: :class:`str`
                Name of coefficient to plot, e.g. ``'CA'``
            *n*: :class:`int`
                Only show the last *n* iterations
            *nAvg*: :class:`int`
                Use the last *nAvg* iterations to compute an average
            *d*: :class:`float`
                Delta in the coefficient to show expected range
            *nLast*: :class:`int`
                Last iteration to use (defaults to last iteration available)
            *nFirst*: :class:`int`
                First iteration to plot
            *FigureWidth*: :class:`float`
                Figure width
            *FigureHeight*: :class:`float`
                Figure height
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: v1.0
            * 2014-12-09 ``@ddalle``: Transferred to :class:`AeroPlot`
            * 2015-02-15 ``@ddalle``: Transferred to :class:`databook.Aero`
            * 2015-03-04 ``@ddalle``: Added *nStart* and *nLast*
            * 2015-12-07 ``@ddalle``: Moved content to base class
        """
        # Plot appropriately.
        return self.PlotValue(c, n=n, **kw)

    # Plot coefficient histogram
    def PlotCoeffHist(self, c: str, nAvg=100, nBin=20, nLast=None, **kw):
        r"""Plot a single coefficient histogram

        :Call:
            >>> h = fm.PlotCoeffHist(comp, c, n=1000, nAvg=100, **kw)
        :Inputs:
            *fm*: :class:`cape.cfdx.databook.CaseFM`
                Instance of the component force history class
            *comp*: :class:`str`
                Name of component to plot
            *c*: :class:`str`
                Name of coefficient to plot, e.g. ``'CA'``
            *nAvg*: :class:`int`
                Use the last *nAvg* iterations to compute an average
            *nBin*: :class:`int`
                Number of bins to plot
            *nLast*: :class:`int`
                Last iteration to use (defaults to last iteration available)
            *FigureWidth*: :class:`float`
                Figure width
            *FigureHeight*: :class:`float`
                Figure height
        :Keyword arguments:
            * See :func:`cape.cfdx.databook.CaseData.PlotValueHist`
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2015-02-15 ``@ddalle``: v1.0
            * 2015-03-06 ``@ddalle``: Added *nLast* and fixed documentation
            * 2015-03-06 ``@ddalle``: Copied to :class:`CaseFM`
        """
        return self.PlotValueHist(c, nAvg=nAvg, nBin=nBin, nLast=None, **kw)
   # >


# Individual component: generic property
class CaseProp(CaseFM):
    pass


# Aerodynamic history class
class CaseResid(CaseData):
    r"""Iterative residual history class

    This class provides an interface to residuals, CPU time, and similar data
    for a given run directory

    :Call:
        >>> hist = cape.cfdx.databook.CaseResid()
    :Outputs:
        *hist*: :class:`cape.cfdx.databook.CaseResid`
            Instance of the run history class
    """
    # Default residual column name
    _default_resid = "L2"

    # Get path to cache file
    def get_cdbfile(self) -> str:
        r"""Get path to iterative history cache file

        :Call:
            >>> cdbfilename = h.get_cdbfile()
        :Inputs:
            *h*: :class:`CaseData`
                Single-case iterative history instance
        :Outputs:
            *cdbfilename*: :class:`str`
                Name of file (extension is ``.cdb``)
        :Versions:
            * 2024-01-22 ``@ddalle``: v1.0
        """
        # Get file name
        return os.path.join("cape", "residual_hist.cdb")

    # Find subiteration start and end iters
    def find_subiters(self, col: str = CASE_COL_SUB_ITERS):
        r"""Find indices of first and last subiter for each whole iter

        :Call:
            >>> maska, maskb = hist.find_subiters(col=None)
        :Inputs:
            *hist*: :class:`CaseResid`
                Iterative residual history instance
            *col*: {``"i_sub"``} | :class:`str`
                Name of subiteration col
        :Outputs:
            *maska*: :class:`np.ndarray`\ [:class:`int`]
                Index of first subiter for each whole iter
            *maskb*: :class:`np.ndarray`\ [:class:`int`]
                Index of last subiter for each whole iter
        :Versions:
            * 2024-07-18 ``@ddalle``: v1.0
        """
        # Default column
        col = CASE_COL_SUB_ITERS if col is None else col
        # Name of subiteration column
        subcol = f"{col}_sub"
        # Check for subiterations
        if subcol not in self:
            return np.zeros(0, "int"), np.zeros(0, "int")
        # Get subiters
        subiters = self[subcol]
        # Calculate floor of each subiteration
        # NOTE: some solvers might use an opposite convention?
        subiter_start = np.floor(subiters)
        # First and last candidate (whole) iteration
        iter_frst = np.min(subiter_start)
        iter_last = np.max(subiter_start)
        # Candidate (whole) iterations
        iters = np.arange(iter_frst, iter_last + 1)
        # Save the iteration at the start of each subiteration
        self.save_col("_subiter_start", subiter_start)
        # Search
        masks, _ = self.find(["_subiter_start"], iters, mapped=True)
        # Remove temporary column
        self.burst_col("_subiter_start")
        # Create arrays for first and last in each iteration
        maska = np.array([mask[0] for mask in masks])
        maskb = np.array([mask[-1] for mask in masks])
        # Limit to cases w/ at least one subiter
        ia = subiters[maska]
        ib = subiters[maskb]
        mask = (ib - ia) >= 0.49
        # Output
        return maska[mask], maskb[mask]

    # Number of orders of magnitude of residual drop
    def GetNOrders(
            self,
            nStats: int = 1,
            col: Optional[str] = None,
            nLast: Optional[int] = None) -> float:
        r"""Get the number of orders of magnitude of residual drop

        :Call:
            >>> nOrders = hist.GetNOrders(nStats=None, col=None)
        :Inputs:
            *hist*: :class:`CaseResid`
                Single-case residual history instance
            *nStats*: {``1``} | :class:`int`
                Number of iters to use for averaging the final residual
            *col*: {None} | :class:`str`
                Name of residual to analyze; default from
                *hist._default_resid*
            *nLast*: {``None``} | :class:`int`
                Last iteration to include in window (default ``-1``)
        :Outputs:
            *nOrders*: {``1``} | :class:`float`
                Number of orders of magnitude of residual drop
        :Versions:
            * 2015-01-01 ``@ddalle``: v1.0
            * 2024-01-24 ``@ddalle``: v2.0; generalize w/ DataKit apprch
            * 2025-08-05 ``@ddalle``: v2.1; add *nLast*
        """
        # Default *col*
        col = self._default_resid if col is None else col
        # Check for *col*
        if col not in self:
            raise KeyError(f"No residual col '{col}' found")
        # Get iters
        iters = self[CASE_COL_ITERS]
        # Check for empty iteration
        if iters.size == 0:
            return np.float64(0.0)
        # Max iteration involved
        imax = np.max(iters)
        # Get last iteration to use
        ib = imax if nLast is None else nLast
        # Check for negative *nLast*
        ib = imax + ib + 1 if ib < 0 else ib
        # Default length of window
        nstats = 1 if nStats is None else abs(nStats)
        # Left-hand side of window
        ia = max(ib - nstats + 1, 0)
        # Identify iterations in window
        mask = (iters >= ia) & (iters <= ib)
        # Get the maximum residual
        L2Max = np.log10(np.max(self[col]))
        # Get the average terminal residual.
        L2End = np.log10(np.mean(self[col][mask]))
        # Return the drop
        return L2Max - L2End

    # Number of orders of unsteady residual drop
    def GetNOrdersUnsteady(self, n=1):
        r"""Get residual drop magnitude

        :Call:
            >>> nOrders = hist.GetNOrders(n=1)
        :Inputs:
            *hist*: :class:`cape.cfdx.databook.CaseResid`
                Instance of the DataBook residual history
            *n*: :class:`int`
                Number of iterations to analyze
        :Outputs:
            *nOrders*: :class:`numpy.ndarray`\ [:class:`float`]
                Number of orders of magnitude of unsteady residual drop
        :Versions:
            * 2015-01-01 ``@ddalle``: First version
        """
        # Process the number of usable iterations available.
        i = max(self.nIter-n, 0)
        # Get the initial residuals
        L1Init = np.log10(self["L1Resid0"][i:])
        # Get the terminal residuals.
        L1End = np.log10(self["L1Resid"][i:])
        # Return the drop
        return L1Init - L1End

    # Plot function
    def PlotResid(self, c='L1Resid', n=None, nFirst=None, nLast=None, **kw):
        r"""Plot a residual by name

        :Call:
            >>> h = hist.PlotResid(c='L1Resid', n=None, **kw)
        :Inputs:
            *hist*: :class:`cape.cfdx.databook.CaseResid`
                Instance of the DataBook residual history
            *c*: :class:`str`
                Name of coefficient to plot
            *n*: :class:`int`
                Only show the last *n* iterations
            *PlotOptions*: :class:`dict`
                Plot options for the primary line(s)
            *nFirst*: :class:`int`
                Plot starting at iteration *nStart*
            *nLast*: :class:`int`
                Plot up to iteration *nLast*
            *FigureWidth*: :class:`float`
                Figure width
            *FigureHeight*: :class:`float`
                Figure height
            *YLabel*: :class:`str`
                Label for *y*-axis
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: v1.0
            * 2014-12-09 ``@ddalle``: v1.1; move to ``AeroPlot``
            * 2015-02-15 ``@ddalle``: v1.2; move to ``databook.Aero``
            * 2015-03-04 ``@ddalle``: v1.3; add *nStart* and *nLast*
            * 2015-10-21 ``@ddalle``: v1.4; from :func:`PlotL1`
            * 2022-01-28 ``@ddalle``: v1.5; add *xcol*
        """
        # Make sure plotting modules are present.
        ImportPyPlot()
        # Initialize dictionary.
        h = {}
        # Iteration field
        xcol = kw.get("xcol", "i")
        xval = self.get_values(xcol)
        # Get iteration numbers.
        if n is None:
            # Use all iterations
            n = xval[-1]
        # Default *nFirst*
        if nFirst is None:
            nFirst = 1
        # Process other options
        fw = kw.get('FigureWidth')
        fh = kw.get('FigureHeight')
        # ---------
        # Last Iter
        # ---------
        # Most likely last iteration
        iB = xval[-1]
        # Check for an input last iter
        if nLast is not None:
            # Attempt to use requested iter.
            if nLast < iB:
                # Using an earlier iter; make sure to use one in the hist.
                jB = self.GetIterationIndex(nLast)
                # Find the iterations that are less than i.
                iB = xval[jB]
        # Get the index of *iB* in *fm.i*.
        jB = np.where(xval == iB)[0][-1]
        # ----------
        # First Iter
        # ----------
        # Get the starting iteration number to use.
        ia = max(xval[0], iB - n + 1, nFirst)
        # Make sure *iA* is in *fm.i* and get the index.
        j0 = self.GetIterationIndex(ia)
        # Reselect *iA* in case initial value was not in *fm.i*.
        ia = int(xval[j0])
        # --------
        # Plotting
        # --------
        # Extract iteration numbers and residuals
        i = xval[j0:]
        # Extract separate first-subiter values
        xcol0 = f"{xcol}_0"
        xval0 = self.get(xcol0)
        # Check if found
        if xval0 is None:
            # Just use *i*, same as main iters
            i0 = i
        else:
            # Filter by *nFirst*
            mask0 = xval0 >= nFirst
            i0 = xval0[mask0]
        # Handling for multiple residuals at same iteration
        di = np.diff(i) != 0
        # First residual at each iteration and last residual at each iteration
        I0 = np.hstack(([True], di))
        I1 = np.hstack((di, [True]))
        # Exclude all *I1* iterations from *I0*
        I0 = np.logical_and(I0, np.logical_not(I1))
        # Nominal residual
        try:
            L1 = self.get_values(c)[j0:]
        except Exception:
            L1 = np.nan*np.ones_like(i)
        # Residual before subiterations
        try:
            # Get values for expected column
            L0 = self.get_values(f'{c}_0')
            # Filter
            if xval0 is None:
                # Filter as if nominal iteration
                L0 = L0[j0:]
            else:
                # Filter using separate i_0 iteration
                L0 = L0[mask0]
        except Exception:
            L0 = np.nan*np.ones_like(i0)
        # Check if L0 is too long.
        if len(L0) > len(i0):
            # Trim it.
            L0 = L0[:len(i0)]
        # Create options
        kw_p = kw.get("PlotOptions", {})
        if kw_p is None:
            kw_p = {}
        kw_p0 = kw.get("PlotOptions0", dict(kw_p))
        if kw_p0 is None:
            kw_p0 = {}
        # Default options
        kw_p0.setdefault("linewidth", 1.2)
        kw_p0.setdefault("color", "b")
        kw_p0.setdefault("linestyle", "-")
        kw_p.setdefault("linewidth", 1.5)
        kw_p.setdefault("color", "k")
        kw_p.setdefault("linestyle", "-")
        # Plot the initial residual if there are any unsteady iterations.
        # (Using specific attribute like "L2Resid_0")
        if L0.size and L0[-1] > L1[-1]:
            h['L0'] = plt.semilogy(i0, L0, **kw_p0)
        # Plot the residual.
        if np.all(I1):
            # Plot all residuals (no subiterations detected)
            h['L1'] = plt.semilogy(i, L1, **kw_p)
        else:
            # Plot first and last subiteration separately
            h['L0'] = plt.semilogy(i[I0], L1[I0], **kw_p0)
            h['L1'] = plt.semilogy(i[I1], L1[I1], **kw_p)
        # Labels
        h['x'] = plt.xlabel('Iteration Number')
        h['y'] = plt.ylabel(kw.get('YLabel', c))
        # Get the figures and axes.
        h['ax'] = plt.gca()
        h['fig'] = plt.gcf()
        # Set figure dimensions
        if fh:
            h['fig'].set_figheight(fh)
        if fw:
            h['fig'].set_figwidth(fw)
        # Attempt to apply tight axes
        _tight_layout()
        # Set the xlimits.
        h['ax'].set_xlim((ia, iB+25))
        # Output.
        return h

    # Plot function
    def PlotL1(self, n=None, nFirst=None, nLast=None, **kw):
        r"""Plot the L1 residual

        :Call:
            >>> h = hist.PlotL1(n=None, nFirst=None, nLast=None, **kw)
        :Inputs:
            *hist*: :class:`cape.cfdx.databook.CaseResid`
                Instance of the DataBook residual history
            *n*: :class:`int`
                Only show the last *n* iterations
            *nFirst*: :class:`int`
                Plot starting at iteration *nStart*
            *nLast*: :class:`int`
                Plot up to iteration *nLast*
            *FigureWidth*: :class:`float`
                Figure width
            *FigureHeight*: :class:`float`
                Figure height
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: v1.0
            * 2014-12-09 ``@ddalle``: v1.1; move to ``AeroPlot``
            * 2015-02-15 ``@ddalle``: v1.2; move to ``databook.Aero``
            * 2015-03-04 ``@ddalle``: v1.3; add *nStart* and *nLast*
            * 2015-10-21 ``@ddalle``: v1.4; refer to ``PlotResid()``
        """
        # Get y-label
        ylbl = kw.get('YLabel', 'L1 Residual')
        # Plot 'L1Resid'
        return self.PlotResid(
            'L1Resid',
            n=n, nFirst=nFirst, nLast=nLast, YLabel=ylbl, **kw)

    # Plot function
    def PlotL2(self, n=None, nFirst=None, nLast=None, **kw):
        r"""Plot the L2 residual

        :Call:
            >>> h = hist.PlotL2(n=None, nFirst=None, nLast=None, **kw)
        :Inputs:
            *hist*: :class:`cape.cfdx.databook.CaseResid`
                Instance of the DataBook residual history
            *n*: :class:`int`
                Only show the last *n* iterations
            *nFirst*: :class:`int`
                Plot starting at iteration *nStart*
            *nLast*: :class:`int`
                Plot up to iteration *nLast*
            *FigureWidth*: :class:`float`
                Figure width
            *FigureHeight*: :class:`float`
                Figure height
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: v1.0
            * 2014-12-09 ``@ddalle``: v1.1; move to ``AeroPlot``
            * 2015-02-15 ``@ddalle``: v1.2; move to ``databook.Aero``
            * 2015-03-04 ``@ddalle``: v1.3; add *nStart* and *nLast*
            * 2015-10-21 ``@ddalle``: v1.4; refer to ``PlotResid()``
        """
        # Get y-label
        ylbl = kw.get('YLabel', 'L2 Residual')
        # Plot 'L2Resid'
        return self.PlotResid(
            'L2Resid', n=n,
            nFirst=nFirst, nLast=nLast, YLabel=ylbl, **kw)

    # Plot function
    def PlotLInf(self, n=None, nFirst=None, nLast=None, **kw):
        r"""Plot the L-infinity residual

        :Call:
            >>> h = hist.PlotLInf(n=None, nFirst=None, nLast=None, **kw)
        :Inputs:
            *hist*: :class:`cape.cfdx.databook.CaseResid`
                Instance of the DataBook residual history
            *n*: :class:`int`
                Only show the last *n* iterations
            *nFirst*: :class:`int`
                Plot starting at iteration *nStart*
            *nLast*: :class:`int`
                Plot up to iteration *nLast*
            *FigureWidth*: :class:`float`
                Figure width
            *FigureHeight*: :class:`float`
                Figure height
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2016-02-04 ``@ddalle``: v1.0
        """
        # Get y-label
        ylbl = kw.get('YLabel', 'L-infinity Residual')
        # Plot 'L1Resid'
        return self.PlotResid(
            'Linf', n=n,
            nFirst=nFirst, nLast=nLast, YLabel=ylbl, **kw)

    # Function to get index of a certain iteration number
    def GetIterationIndex(self, i):
        r"""Return index of a particular iteration in *hist.i*

        If the iteration *i* is not present in the history, the index of the
        last available iteration less than or equal to *i* is returned.

        :Call:
            >>> j = hist.GetIterationIndex(i)
        :Inputs:
            *hist*: :class:`cape.cfdx.databook.CaseResid`
                Instance of the residual history class
            *i*: :class:`int`
                Iteration number
        :Outputs:
            *j*: :class:`int`
                Index of last iteration in *fm.i* less than or equal to *i*
        :Versions:
            * 2015-03-06 ``@ddalle``: v1.0
            * 2024-01-10 ``@ddalle``: v1.1; DataKit updates
        """
        # Get iterations
        iters = self.get_values("i")
        # Check for *i* less than first iteration.
        if i < iters[0]:
            return 0
        # Find the index.
        j = np.where(iters <= i)[0][-1]
        # Output
        return j


# Individual component time series force and moment
class CaseTS(CaseFM):
    r"""Force and moment time series iterative histories

    This class contains methods for reading data about an the histroy of
    an individual component for a single case.

    :Call:
        >>> fm = cape.cfdx.databook.CaseFM(C, MRP=None, A=None)
    :Inputs:
        *C*: :class:`list` (:class:`str`)
            List of coefficients to initialize
        *MRP*: :class:`numpy.ndarray`\ [:class:`float`] shape=(3,)
            Moment reference point
        *A*: :class:`numpy.ndarray` shape=(*N*,4) or shape=(*N*,7)
            Matrix of forces and/or moments at *N* iterations
    :Outputs:
        *fm*: :class:`cape.aero.FM`
            Instance of the force and moment class
        *fm.coeffs*: :class:`list` (:class:`str`)
            List of coefficients
        *fm.MRP*: :class:`numpy.ndarray`\ [:class:`float`] shape=(3,)
            Moment reference point
    :Versions:
        * 2024-10-09 ``@aburkhea``: Started
    """
   # --- Class attributes ---
    # Attributes
    __slots__ = (
        "comp",
    )

    # Minimal list of columns
    _base_cols = (
        "i",
        "solver_iter",
        "t",
        "solver_time",
        "CA",
        "CY",
        "CN",
        "CLL",
        "CLM",
        "CLN",
    )
    # Minimal list of "coeffs"
    _base_coeffs = (
        "CA",
        "CY",
        "CN",
        "CLL",
        "CLM",
        "CLN",
    )

   # --- Write ---
    # Write to cape db file
    def write_dbook_cdb(self, fname):
        r"""Write contents of history to ``.cdb`` file

        :Call:
            >>> fm.write_dbook_cdb(fname)
        :Inputs:
            *fm*: :class:`CaseData`
                Iterative history instance
        :Versions:
            * 2024-10-09 ``@aburkhea``: v1.0
        """
        # Try to write it
        try:
            # Create database
            db = capefile.CapeFile(self)
            # Write file
            db.write(fname)
        except PermissionError:
            print(f"    Lacking permissions to write '{fname}'")

   # --- Data ---
    # Get end time
    def get_tend(self, tcol):
        r"""Get end time in time series case data

        :Call:
            >>> fm.get_tend
        :Inputs:
            *fm*: :class:`CaseData`
                Iterative history instance
        :Versions:
            * 2024-10-10 ``@aburkhea``: v1.0
        """
        # Get last and max tcol index
        nEnd = len(self[tcol]) - 1
        nMax = np.argmax(self[tcol])
        # Last time should be largest
        tEnd = self[tcol][-1] if nEnd == nMax else None
        return tEnd


# Set font
def _set_font(h=None):
    r"""Set font family of a Matplotlib text object

    When this function is called for the first time, it searches for
    which fonts are available and picks the most favorable.

    :Versions:
        * 2024-01-22 ``@ddalle``: v1.0
        * 2024-05-16 ``@ddalle``: v1.1; allow 0-arg call
    """
    # Check if font families cached
    if len(FONT_FAMILY) == 0:
        # Import font manager
        from matplotlib import font_manager
        # Initialize fonts
        fontnames = []
        # Loop through font file names
        for ffont in font_manager.findSystemFonts():
            # Try to get the name of the font based on file name
            try:
                fontname = font_manager.FontProperties(fname=ffont).get_name()
            except RuntimeError:
                continue
            # Append to font name list
            fontnames.append(fontname)
        # Loop through candidates
        for family in _FONT_FAMILY:
            if family in fontnames:
                FONT_FAMILY.append(family)
        # Add "sans-serif" fallback
        FONT_FAMILY.append("sans-serif")
    # Exit if no input
    if h is None:
        return
    # Use a fixed set of families
    h.set_family(FONT_FAMILY)


def _mask_repeat_iters(iters: np.ndarray) -> np.ndarray:
    r"""Get mask of iterations to keep after imperfect restart

    If a previous case continues past its last restart, the history may
    contain some iterations that get overwritten during the next run.
    This function returns a mask of iters to keep.

    :Call:
        >>> mask = _mask_repeat_ters(iters)
    :Inputs:
        *iters*: :class:`np.ndarray`
            Iteration numbers
    :Outputs:
        *mask*: :class:`np.ndarray`\ [:class:`bool`]
            Mask of which iterations to keep, eliminating repeats
    """
    # Check for empty array
    if iters.size == 0:
        return np.ones(0, dtype="bool")
    # For each iter, calculate minimum of all iters after
    imin_r = _cummin_r(iters)
    # Shift by one
    imin_r_shift = np.hstack((imin_r[1:], imin_r[-1] + 1))
    # Only keep iters who are strictly less than min of following iters
    return iters < imin_r_shift


def _cummin_r(arr: np.ndarray) -> np.ndarray:
    r"""Calculate reversed cumulative minimum of a 1D array

    :Call:
        >>> v = _cummin_r(arr)
    """
    # Calculate cumulative minimum in revers
    return np.flip(np.minimum.accumulate(np.flip(arr)))


# Apply built-in tight_layout() function
def _tight_layout():
    ImportPyPlot()
    try:
        plt.tight_layout()
    except Exception:  # pragma no cover
        pass
