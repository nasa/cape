#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.cfdx.databook`: Post-processed data module
=====================================================

This module contains functions for reading and processing forces,
moments, and other entities from cases in a trajectory.  This module
forms the core for all database post-processing in Cape, but several
other database modules exist for more specific applications:

    * :mod:`cape.cfdx.lineload`
    * :mod:`cape.cfdx.pointsensor`

This module provides three basic classes upon which more specific data
classes are developed:

    * :class:`DataBook`: Overall databook container
    * :class:`DataBookComp`: Template databook for an individual component
    * :class:`CaseData`: Template class for one case's iterative history

The first two of these are subclassed from :class:`dict`, so that
generic data can be accessed with syntax such as ``DB[coeff]`` for an
appropriately named coefficient.  An outline of derived classes for
these three templates is shown below.

    * :class:`DataBook`
        - :class:`TriqFMDataBook`: post-processed forces & moments

    * :class:`DataBookComp`
        - :class:`FMDataBook`: force & moment data, one comp
        - :class:`TargetDataBook`: target data
        - :class:`TriqFMFaceDataBook`: surface CP FM for one comp
        - :class:`LineLoadDataBook`: sectional load databook
        - :class:`PointSensorGroupDataBook`: group of points
        - :class:`TriqPointGroupDataBook`: group of surface points
        - :class:`PointSensorDataBook`: one point sensor
        - :class:`TriqPointDataBook`: one surface point sensor

    * :class:`CaseData`
        - :class:`CaseFM`: iterative force & moment history
        - :class:`CaseResid`: iterative residual history

In addition, each solver has its own version of this module:

    * :mod:`cape.pycart.dataBook`
    * :mod:`cape.pyfun.dataBook`
    * :mod:`cape.pyover.dataBook`

The parent class :class:`cape.cfdx.databook.DataBook` provides a common
interface to all of the requested force, moment, point sensor, etc.
quantities that have been saved in the data book. Informing :mod:`cape`
which quantities to track, and how to statistically process them, is
done using the ``"DataBook"`` section of the JSON file, and the various
data book options are handled within the API using the
:mod:`cape.cfdx.options.DataBook` module.

The master data book class :class:`cape.cfdx.databook.DataBook` is based
on the built-in :class:`dict` class with keys pointing to force and
moment data books for individual components. For example, if the JSON
file tells Cape to track the forces and/or moments on a component called
``"body"``, and the data book is the variable *DB*, then the forces and
moment data book is ``DB["body"]``.  This force and moment data book
contains statistically averaged forces and moments and other statistical
quantities for every case in the run matrix. The class of the force and
moment data book is :class:`cape.cfdx.databook.FMDataBook`.

The data book also has the capability to store "target" data books so
that the user can compare results of the current CFD solutions to
previous results or experimental data. These are stored in
``DB["Targets"]`` and use the :class:`cape.cfdx.databook.TargetDataBook`
class. Other types of data books can also be created, such as the
:class:`cape.cfdx.pointsensor.PointSensorDataBook` class for tracking
statistical properties at individual points in the solution field. Data
books for tracking results of groups of cases are built off of the
:class:`cape.cfdx.databook.DataBookComp` class, which contains many common
tools such as plotting.

The :mod:`cape.cfdx.dataBook` module also contains modules for
processing results within individual case folders. This includes the
:class:`cape.cfdx.databook.CaseFM` module for reading iterative
force/moment histories and the :class:`cape.cfdx.databook.CaseResid`
for iterative histories of residuals.

"""

# Standard library modules
import importlib
import os
import sys
import time
import traceback
import warnings
from typing import Optional
from abc import abstractmethod

# Third-party modules
import numpy as np

# Local modules
from . import casecntl
from .. import pltfile
from .. import trifile
from .. import util
from ..dkit.rdb import DataKit
from .casedata import (
    CaseFM,
    CaseProp,
    CaseTS,
    CaseResid,
    DBPlotOpts,
    FONT_FAMILY,
    _set_font,
    _tight_layout,
)
from .databookbase import DataBookBase


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


# Suppress warnings
warnings.simplefilter("ignore")


def _set_agg():
    # Check compatibility of the environment
    if os.environ.get('DISPLAY') is None:
        # Use a special MPL backend to avoid need for DISPLAY
        import matplotlib
        matplotlib.use('Agg')


def _import_mplmod(modname: str):
    # Check if present
    mod = sys.modules.get(modname)
    # Return if if found
    if mod is not None:
        return mod
    # Otherwise set back-end
    _set_agg()
    # Import the module
    return importlib.import_module(modname)


def import_pyplot():
    return _import_mplmod("matplotlib.pyplot")


# Function to automatically get inclusive data limits.
def get_ylim(ha, pad=0.05):
    r"""Calculate appropriate *y*-limits to include all lines in a plot

    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` and
    :class:`matplotlib.collections.PolyCollection` are checked.

    :Call:
        >>> ymin, ymax = get_ylim(ha, pad=0.05)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *pad*: :class:`float`
            Extra padding to min and max values to plot.
    :Outputs:
        *ymin*: :class:`float`
            Minimum *y* coordinate including padding
        *ymax*: :class:`float`
            Maximum *y* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: v1.0
    """
    # Initialize limits.
    ymin = np.inf
    ymax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Get the y data for this line
            ydata = h.get_ydata()
            # Check the min and max data
            if len(ydata) > 0:
                ymin = min(ymin, min(h.get_ydata()))
                ymax = max(ymax, max(h.get_ydata()))
        elif t in ['PathCollection', 'PolyCollection']:
            # Loop through paths
            for P in h.get_paths():
                # Get the coordinates
                ymin = min(ymin, min(P.vertices[:, 1]))
                ymax = max(ymax, max(P.vertices[:, 1]))
    # Check for identical values
    if ymax - ymin <= 0.1*pad:
        # Expand by manual amount,.
        ymax += pad*abs(ymax)
        ymin -= pad*abs(ymin)
    # Add padding.
    yminv = (1+pad)*ymin - pad*ymax
    ymaxv = (1+pad)*ymax - pad*ymin
    # Output
    return yminv, ymaxv


# Function to automatically get inclusive data limits.
def get_xlim(ha, pad=0.05):
    r"""Calculate appropriate *x*-limits to include all lines in a plot

    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` are
    checked.

    :Call:
        >>> xmin, xmax = get_xlim(ha, pad=0.05)
    :Inputs:
        *ha*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *pad*: :class:`float`
            Extra padding to min and max values to plot.
    :Outputs:
        *xmin*: :class:`float`
            Minimum *x* coordinate including padding
        *xmax*: :class:`float`
            Maximum *x* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: v1.0
    """
    # Initialize limits.
    xmin = np.inf
    xmax = -np.inf
    # Loop through all children of the input axes.
    for h in ha.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Get data
            xdata = h.get_xdata()
            # Check the min and max data
            if len(xdata) > 0:
                xmin = min(xmin, min(h.get_xdata()))
                xmax = max(xmax, max(h.get_xdata()))
        elif t in ['PathCollection', 'PolyCollection']:
            # Loop through paths
            for P in h.get_paths():
                # Get the coordinates
                xmin = min(xmin, min(P.vertices[:, 1]))
                xmax = max(xmax, max(P.vertices[:, 1]))
    # Check for identical values
    if xmax - xmin <= 0.1*pad:
        # Expand by manual amount,.
        xmax += pad*abs(xmax)
        xmin -= pad*abs(xmin)
    # Add padding.
    xminv = (1+pad)*xmin - pad*xmax
    xmaxv = (1+pad)*xmax - pad*xmin
    # Output
    return xminv, xmaxv


# Data book for an individual component
class DataBookComp(DataKit):
    r"""Individual item data book basis class

    :Call:
        >>> DBi = DataBookComp(comp, cntl, check=False, lock=False)
    :Inputs:
        *comp*: :class:`str`
            Name of the component or other item name
        *cntl*: :class:`Cntl`
            CAPE control class instance
        *check*: ``True`` | {``False``}
            Whether or not to check LOCK status
        *lock*: ``True`` | {``False``}
            If ``True``, wait if the LOCK file exists
    :Outputs:
        *DBi*: :class:`cape.cfdx.databook.DataBookComp`
            An individual item data book
    :Versions:
        * 2014-12-22 ``@ddalle``: v1.0
        * 2015-12-04 ``@ddalle``: v1.0 (fork :class:`DBComp`)
        * 2025-01-22 ``@aburkhea``: v2.0
        * 2025-05-25 ``@ddalle``: v2.1; rename DBBase -> DataBookComp
    """
  # ======
  # Config
  # ======
  # <
    # Initialization method
    def __init__(self, comp, cntl, check=False, lock=False, **kw):
        """Initialization method

        :Versions:
            * 2014-12-21 ``@ddalle``: v1.0
            * 2016-03-15 ``@ddalle``: v1.1; general column names
            * 2022-04-13 ``@ddalle``: v1.2; use *cntl*
        """
        # Unpack
        x = cntl.x
        opts = cntl.opts
        # Save relevant inputs
        self.x = x.Copy()
        self.opts = opts
        self.cntl = cntl
        self.comp = comp
        self.name = comp
        self.sources = {}
        # Root directory
        self.RootDir = kw.get("RootDir", os.getcwd())

        # Get the directory.
        fdir = opts.get_DataBookFolder()

        # Construct the file name.
        fcomp = f'aero_{comp}.csv'
        # Folder name for compatibility.
        fdir = fdir.replace("/", os.sep)
        fdir = fdir.replace("\\", os.sep)
        # Construct the full file name.
        fname = os.path.join(fdir, fcomp)
        # Save the file name.
        self.fname = fname
        self.fdir = fdir

        # Create directories if necessary
        if not os.path.isdir(fdir):
            # Create data book folder (should not occur)
            os.mkdir(fdir)

        # Read the file or initialize empty arrays.
        self.Read(self.fname, check=check, lock=lock)

    # Command-line representation
    def __repr__(self):
        r"""Representation method

        :Versions:
            * 2014-12-27 ``@ddalle``: v1.0
        """
        # Initialize string
        try:
            return "<DataBookComp '%s', n=%s>" % (self.comp, self.n)
        except Exception:
            return "<DataBookComp, n=%i>" % self.n
    # String conversion
    __str__ = __repr__

    # Directory creation using appropriate settings
    def mkdir(self, fdir):
        r"""Create a directory using settings from *DataBook>umask*

        :Call:
            >>> DB.mkdir(fdir)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the Cape data book class
            *fdir*: :class:`str`
                Directory to create
        :Versions:
            * 2017-09-05 ``@ddalle``: v1.0
        """
        # Call databook method
        os.mkdir(fdir)
  # >

  # ======
  # Read
  # ======
  # <

    # Process FM data book columns
    def ProcessColumns(self):
        r"""Process column names

        :Call:
            >>> DBi.ProcessColumns()
        :Inputs:
            *DBi*: :class:`cape.cfdx.databook.DataBookComp`
                Data book base object
        :Effects:
            *DBi.xCols*: :class:`list` (:class:`str`)
                List of trajectory keys
            *DBi.fCols*: :class:`list` (:class:`str`)
                List of floating point data columns
            *DBi.iCols*: :class:`list` (:class:`str`)
                List of integer data columns
            *DBi.cols*: :class:`list` (:class:`str`)
                Total list of columns
            *DBi.nxCol*: :class:`int`
                Number of trajectory keys
            *DBi.nfCol*: :class:`int`
                Number of floating point keys
            *DBi.niCol*: :class:`int`
                Number of integer data columns
            *DBi.nCol*: :class:`int`
                Total number of columns
        :Versions:
            * 2016-03-15 ``@ddalle``: v1.0
        """
        # Get coefficients
        coeffs = self.opts.get_DataBookCols(self.comp)
        # Initialize columns for coefficients
        cCols = []
        # Check for mean
        for coeff in coeffs:
            # Get list of stats for this column
            cColi = self.opts.get_DataBookColStats(self.comp, coeff)
            # Check for 'mu'
            if 'mu' in cColi:
                cCols.append(coeff)
        # Add list of statistics for each column
        for coeff in coeffs:
            # Get list of stats for this column
            cColi = self.opts.get_DataBookColStats(self.comp, coeff)
            # Remove 'mu' from the list
            if 'mu' in cColi:
                cColi.remove('mu')
            # Append to list
            for c in cColi:
                cCols.append('%s_%s' % (coeff, c))
        # Get additional float columns
        fCols = self.opts.get_DataBookFloatCols(self.comp)
        iCols = self.opts.get_DataBookIntCols(self.comp)

        # Save column names.
        self.xCols = self.x.cols
        self.fCols = cCols + fCols
        self.iCols = iCols
        self.cols = self.xCols + self.fCols + self.iCols
        # Counts
        self.nxCol = len(self.xCols)
        self.nfCol = len(self.fCols)
        self.niCol = len(self.iCols)
        self.nCol = len(self.cols)

    # Set converters
    def ProcessConverters(self):
        r"""Process the list of converters to read and write each column

        :Call:
            >>> DBP.ProcessConverters()
        :Inputs:
            *DBP*: :class:`DataBookBase`
                Data book base object
        :Effects:
            *DBP.rconv*: :class:`list` (:class:`function`)
                List of read converters
            *DBP.wflag*: :class:`list` (%i | %.12g | %s)
                List of write flags
        :Versions:
            * 2016-03-15 ``@ddalle``: v1.0
        """
        # List of converters
        self.rconv = []
        self.wflag = []
        # Initialize trajectory columns
        for k in self.xCols:
            # Get the type
            t = self.x.defns[k].get('Value', 'float')
            # Set the converter
            if t == 'float':
                # Float value
                self.rconv.append(float)
                self.wflag.append('%.12g')
            elif t == 'int':
                # Regular integer value
                self.rconv.append(int)
                self.wflag.append('%i')
            elif t in ['str']:
                # String value
                self.rconv.append(str)
                self.wflag.append('%s')
            elif t in ['bytes']:
                # Unicode string
                self.rconv.append(bytes)
                self.wflag.append('%s')
            elif t in ['oct', 'octal']:
                # Octal integer
                self.rconv.append(lambda v: eval('0o'+v))
                self.wflag.append('%i')
            elif t in ['bin', 'binary']:
                # Binary integer
                self.rconv.append(lambda v: eval('0b'+v))
                self.wflag.append('%i')
            elif t in ['hex']:
                # Hexadecimal integer
                self.rconv.append(lambda v: eval('0x'+v))
                self.wflag.append('%i')
        # Initialize float columns
        for k in self.fCols:
            self.rconv.append(float)
            self.wflag.append('%.12g')
        # Initialize int columns
        for k in self.iCols:
            self.rconv.append(int)
            self.wflag.append('%.12g')

    @abstractmethod
    def ReadRawData(self, comp):
        pass

    @abstractmethod
    def ReadCase(self):
        """ Read data book
        """
        pass

    # Read databook summary file
    def Read(self, fname, check=False, lock=False):
        r"""Read a data book statistics file

        :Call:
            >>> DBc.Read()
            >>> DBc.Read(fname, check=False, lock=False)
        :Inputs:
            *DBc*: :class:`cape.cfdx.databook.DataBookComp`
                Data book base object
            *fname*: :class:`str`
                Name of data file to read
            *check*: ``True`` | {``False``}
                Whether or not to check LOCK status
            *lock*: ``True`` | {``False``}
                If ``True``, wait if the LOCK file exists
        :Versions:
            * 2015-12-04 ``@ddalle``: v1.0
            * 2017-06-12 ``@ddalle``: Added *lock*
        """
        # Check for lock status?
        if check:
            # Wait until unlocked
            while self.CheckLock():
                # Status update
                print("   Locked.  Waiting 30 s ...")
                os.sys.stdout.flush()
                time.sleep(30)
        # Lock the file?
        if lock:
            self.Lock()
        # Read the file if it exists
        if os.path.isfile(fname):
            self.read_csv(fname)
            # Force set number of rows
            if self.cols:
                self.n = len(self[self.cols[0]])
            else:
                self.n = 0
            # Convert lists to np.array for bwds compatibility
            if self.cols:
                for k in self.cols:
                    if isinstance(self[k], list):
                        # Initialize an empty array.
                        self[k] = np.array(self[k], dtype=str)
            # Force iCols to int
            for icol in self.iCols:
                v = self.burst_col(icol)
                self.save_col(icol, v.astype(np.int32))
        else:
            # Initialize empty trajectory arrays
            for k in self.xCols:
                # get the type.
                t = self.x.defns[k].get('Value', 'float')
                # convert type
                if t in ['hex', 'oct', 'octal', 'bin']:
                    t = 'int'
                # Initialize an empty array.
                self.save_col(k, np.array([], dtype=str(t)))
            # Initialize float parameters
            for col in self.fCols:
                self.save_col(col, np.array([], dtype=float))
            # Initialize integer counts
            for col in self.iCols:
                self.save_col(col, np.array([], dtype=int))
            # Exit
            self.n = 0
            return

    # Read a copy of databook summary file
    def ReadCopy(self, check=False, lock=False):
        r"""Read a copied database object

        :Call:
            >>> DBc1 = DBc.ReadCopy(check=False, lock=False)
        :Inputs:
            *DBc*: :class:`cape.cfdx.databook.DataBookComp`
                Data book base object
            *check*: ``True`` | {``False``}
                Whether or not to check LOCK status
            *lock*: ``True`` | {``False``}
                If ``True``, wait if the LOCK file exists
        :Outputs:
            *DBc1*: :class:`cape.cfdx.databook.DataBookComp`
                Copy of data book base object
        :Versions:
            * 2017-06-26 ``@ddalle``: v1.0
        """
        # Check for a name
        try:
            # Use the *name* as the first choice
            name = self.name
        except AttributeError:
            # Fall back to the *comp* attribute
            name = self.comp
        # Call the object
        DBc = self.__class__(name, self.cntl, check=check, lock=lock)
        # Ensure the same root directory is used
        DBc.RootDir = getattr(self, "RootDir", os.getcwd())
        # Output
        return DBc

  # >

  # ----
  # Lock
  # ----
  # <
  # [
    # Get name of lock file
    def GetLockFile(self):
        r"""Get the name of the potential lock file

        :Call:
            >>> flock = DBc.GetLockFile()
        :Inputs:
            *DBc*: :class:`cape.cfdx.databook.DataBookBase`
                Data book base object
        :Outputs:
            *flock*: :class:`str`
                Full path to potential ``lock`` file
        :Versions:
            * 2017-06-12 ``@ddalle``: v1.0
        """
        # Split file name so we can insert "lock." at the right place
        fdir, fn = os.path.split(self.fname)
        # Construct lock file name
        flock = os.path.join(fdir, "lock.%s" % fn)
        # Check for absolute path
        if not os.path.isabs(flock):
            # Append root directory
            flock = os.path.join(self.RootDir, flock)
        # Output
        return flock

    # Check lock file
    def CheckLock(self):
        r"""Check if lock file for this component exists

        :Call:
            >>> q = DBc.CheckLock()
        :Inputs:
            *DBc*: :class:`cape.cfdx.databook.DataBookBase`
                Data book base object
        :Outputs:
            *q*: :class:`bool`
                Whether or not corresponding LOCK file exists
        :Versions:
            * 2017-06-12 ``@ddalle``: v1.0
        """
        # Get the name of the lock file
        flock = self.GetLockFile()
        # Check if the file exists
        if os.path.isfile(flock):
            # Get the mod time of said file
            tlock = os.path.getmtime(flock)
            # Check for a stale file (using 1.5 hrs)
            if time.time() - tlock > 5400.0:
                # Stale file; not locked
                try:
                    os.remove(flock)
                except Exception:
                    pass
                return False
            else:
                # Still locked
                return True
        else:
            # File does not exist
            return False

    # Write the lock file
    def Lock(self):
        """Write a 'LOCK' file for a data book component

        :Call:
            >>> DBc.Lock()
        :Inputs:
            *DBc*: :class:`cape.cfdx.databook.DataBookBase`
                Data book base object
        :Versions:
            * 2017-06-12 ``@ddalle``: v1.0
        """
        # Name of the lock file
        flock = self.GetLockFile()
        # Safely lock
        try:
            # Open the file
            f = open(flock, 'w')
            # Write the name of the component.
            try:
                f.write("%s\n" % self.comp)
            except AttributeError:
                pass
            # Close the file
            f.close()
        except Exception:
            pass

    # Touch the lock file
    def TouchLock(self):
        """Touch a 'LOCK' file for a data book component to reset its mod time

        :Call:
            >>> DBc.TouchLock()
        :Inputs:
            *DBc*: :class:`cape.cfdx.databook.DataBookBase`
                Data book base object
        :Versions:
            * 2017-06-14 ``@ddalle``: v1.0
        """
        # Name of the lock file
        flock = self.GetLockFile()
        # Update the file
        os.utime(flock, None)

    # Unlock the file
    def Unlock(self):
        """Delete the LOCK file if it exists

        :Call:
            >>> DBc.Unlock()
        :Inputs:
            *DBc*: :class:`cape.cfdx.databook.DataBookBase`
                Data book base object
        :Versions:
            * 2017-06-12 ``@ddalle``: v1.0
        """
        # Name of the lock file
        flock = self.GetLockFile()
        # Check if it exists
        if os.path.isfile(flock):
            # Delete the file
            os.remove(flock)
  # ]
  # >

  # ======
  # Write
  # ======
  # <
    @abstractmethod
    def UpdateCaseDB(self):
        pass

    @abstractmethod
    def DeleteCasesDB(self):
        pass

    # Function to delete entries by index
    def DeleteCases(self, I, comp):
        r"""Delete list of cases from data book

        :Call:
            >>> n = DB.Delete(I)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the pyCart data book class
            *I*: :class:`list`\ [:class:`int`]
                List of trajectory indices
        :Outputs:
            *n*: :class:`int`
                Number of deleted entries
        :Versions:
            * 2015-03-13 ``@ddalle``: v1.0
            * 2017-04-13 ``@ddalle``: Split by component
        """
        # Get the first data book component.
        DBc = self
        # Number of cases in current data book.
        nCase = DBc.n
        # Initialize data book index array.
        J = []
        # Loop though indices to delete.
        for i in I:
            # Find the match.
            j = DBc.FindMatch(i)
            # Check if one was found.
            if j is None:
                continue
            # Append to the list of data book indices.
            J.append(j)
        # Number of deletions
        nj = len(J)
        # Exit if no deletions
        if nj == 0:
            return nj

        # Report status
        print("  Removing %s entries from FM component '%s'" % (nj, comp))
        # Initialize mask of cases to keep.
        mask = np.ones(nCase, dtype=bool)
        # Set values equal to false for cases to be deleted.
        mask[J] = False
        # Loop through data book columns.
        for c in DBc.keys():
            # Apply the mask
            DBc[c] = DBc[c][mask]
        # Update the number of entries.
        DBc.n = len(DBc[c])
        # Output
        return nj

    # Update case in data book
    def UpdateCase(self, i, comp):
        # Get the first data book component.
        DBc = self
        # Try to find a match existing in the data book.
        j = DBc.FindMatch(i)
        # Get the name of the folder.
        frun = self.x.GetFullFolderNames(i)
        # Status update.
        print(frun)
        # Go home.
        os.chdir(self.RootDir)
        # Check if the folder exists.
        if not os.path.isdir(frun):
            # Nothing to do.
            return 0
        # Go to the folder.
        os.chdir(frun)
        # Get the current iteration number
        nIter = self.cntl.GetCurrentIter(i)
        # Get the number of iterations used for statutils.
        nStats = self.opts.get_DataBookNStats(comp)
        # Get the iteration at which statistics can begin.
        nMin = self.opts.get_DataBookNMin(comp)
        # Process whether or not to update.
        if (not nIter) or (nIter < nMin + nStats):
            # Not enough iterations (or zero iterations)
            print("  Not enough iterations (%s) for analysis." % nIter)
            q = False
        elif j is None:
            # No current entry.
            print("  Adding new databook entry at iteration %i." % nIter)
            q = True
        elif DBc['nIter'][j] < nIter:
            # Update
            print(
                "  Updating from iteration %i to %i."
                % (DBc['nIter'][j], nIter))
            q = True
        elif DBc['nStats'][j] < nStats:
            # Change statistics
            print("  Recomputing statistics using %i iterations." % nStats)
            q = True
        else:
            # Up-to-date
            print("  Databook up to date.")
            q = False
        # Check for an update
        if (not q):
            return 0
        # Call specific databook updater
        self.UpdateCaseDB(i, j, comp)
        # Go back.
        os.chdir(self.RootDir)
        # Output
        return 1

    # Write data book
    def Write(self, fname=None, merge=False, unlock=True):
        """Write a single data book summary file

        :Call:
            >>> DBi.WriteDB()
            >>> DBi.WriteDB(fname, merge=False, unlock=True)
        :Inputs:
            *DBi*: :class:`cape.cfdx.databook.DataBookComp`
                An individual item data book
            *fname*: :class:`str`
                Name of data file to read
            *merge*: ``True`` | {``False``}
                Whether or not to attempt a merger before writing
            *unlock*: {``True``} | ``False``
                Whether or not to delete any lock files
        :Versions:
            * 2015-12-04 ``@ddalle``: v1.0
            * 2017-06-12 ``@ddalle``: Added *unlock*
            * 2017-06-26 ``@ddalle``: Added *merge*
        """
        # Check merger option
        if merge:
            # Read a copy
            DBc = self.ReadCopy(check=True, lock=True)
            # Merge it
            self.Merge(DBc)
            # Re-sort
            self.Sort()
        # Check for default file name
        if fname is None:
            fname = self.fname
        # check for a previous old file.
        if os.path.isfile(fname + ".old"):
            # Remove it
            os.remove(fname + ".old")
        # Check for an existing data file.
        if os.path.isfile(fname):
            # Move it to ".old"
            os.rename(fname, fname + ".old")
        # Write dbook to f
        self.write_csv_dense(fname)
        # Unlock
        if unlock:
            self.Unlock()
  # >

  # ==============
  # Organization
  # ==============
  # <

    # Match the databook copy of the trajectory
    def UpdateRunMatrix(self):
        """Match the trajectory to the cases in the data book

        :Call:
            >>> DBi.UpdateRunMatrix()
        :Inputs:
            *DBi*: :class:`cape.cfdx.databook.DataBookComp`
                Component data book
        :Versions:
            * 2017-04-18 ``@ddalle``: v1.0
        """
        # Copy trajectory
        self.x = self.x.Copy()
        # Loop through the fields.
        for k in self.x.cols:
            # Copy the data.
            if k in self:
                # Copy the data
                self.x[k] = self[k]
                # Set the text.
                self.x.text[k] = [str(xk) for xk in self[k]]
            else:
                # Set empty data
                self.x[k] = np.nan*np.ones(self.n)
                # Empty text
                self.x.text[k] = ["" for k in range(self.n)]
        # Set the number of cases.
        self.x.nCase = self.n

    # Merge another copy
    def Merge(self, DBc):
        r"""Merge another copy of the data book object

        :Call:
            >>> DBi.Merge(DBc)
        :Inputs:
            *DBi*: :class:`cape.cfdx.databook.DataBookComp`
                Component data book
            *DBc*: :class:`cape.cfdx.databook.DataBookComp`
                Copy of component data book, perhaps read at a different time
        :Versions:
            * 2017-06-26 ``@ddalle``: v1.0
        """
        # Check for consistency
        if self.cols != DBc.cols:
            raise KeyError("Data book objects do not have same list of keys")
        # Loop through the entries of *DBc*
        for j in range(DBc.n):
            # Check for matches
            i = DBc.FindDBMatch(self, j)
            # Check for a match
            if i is not None:
                # Check for iteration
                if 'nIter' not in self:
                    # No decider
                    continue
                elif self['nIter'][i] >= DBc['nIter'][j]:
                    # Current one is newer (or tied)
                    continue
                else:
                    # *DBc* has newer value
                    for k in self.cols:
                        self[k][i] = DBc[k][j]
                    # Avoid n+=1 counter
                    continue
            # No matches; merge
            for k in self.cols:
                self[k] = np.append(self[k], DBc[k][j])
            # Increase count
            self.n += 1
        # Sort
        self.Sort()

    # Function to get sorting indices.
    def ArgSort(self, key=None):
        r"""Return indices that would sort a data book by a trajectory key

        :Call:
            >>> I = DBi.ArgSort(key=None)
        :Inputs:
            *DBi*: :class:`cape.cfdx.databook.DataBookComp`
                An individual item data book
            *key*: :class:`str`
                Name of trajectory key to use for sorting; default is first key
        :Outputs:
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indices; must have same size as data book
        :Versions:
            * 2014-12-30 ``@ddalle``: v1.0
        """
        # Process the key.
        if key is None:
            key = self.x.cols[0]
        # Check for multiple keys.
        if type(key).__name__ in ['list', 'ndarray', 'tuple']:
            # Init pre-array list of ordered n-lets like [(0, 1, 0), ..., ]
            Z = zip(*[self[k] for k in key])
            # Init list of key definitions
            dt = []
            # Loop through keys to get data types (dtype)
            for k in key:
                # Get the type.
                dtk = self.x.defns[k]['Value']
                # Convert it to numpy jargon.
                if dtk in ['float']:
                    # Numeric value
                    dt.append((str(k), 'f'))
                elif dtk in ['int', 'hex', 'oct', 'octal']:
                    # Stored as an integer
                    dt.append((str(k), 'i'))
                else:
                    # String is default.
                    dt.append((str(k), 'S32'))
            # Create the array to be used for multicolumn sort.
            A = np.array(Z, dtype=dt)
            # Get the sorting order
            I = np.argsort(A, order=[str(k) for k in key])
        else:
            # Indirect sort on a single key.
            I = np.argsort(self[key])
        # Output.
        return I

    # Function to sort data book
    def Sort(self, key=None, I=None):
        r"""Sort a data book according to either a key or an index

        :Call:
            >>> DBi.Sort()
            >>> DBi.Sort(key)
            >>> DBi.Sort(I=None)
        :Inputs:
            *DBi*: :class:`cape.cfdx.databook.DataBookComp`
                An individual item data book
            *key*: :class:`str`
                Name of trajectory key to use for sorting; default is first key
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indices; must have same size as data book
        :Versions:
            * 2014-12-30 ``@ddalle``: v1.0
            * 2017-04-18 ``@ddalle``: Using :func:`np.lexsort`
        """
        # Process inputs.
        if I is not None:
            # Index array specified; check its quality.
            if type(I).__name__ not in ["ndarray", "list"]:
                # Not a suitable list.
                raise TypeError("Index list is unusable type.")
            elif len(I) != self.n:
                # Incompatible length.
                raise IndexError(
                    ("Index list length (%i) " % len(I)) +
                    ("is not equal to data book size (%i)." % self.n))
        elif key is not None:
            # Use ArgSort to get indices that sort on that key.
            I = self.ArgSort(key)
        else:
            # List of x variables
            try:
                # There should be a list if we weren't lasy in __init__
                xkeys = self.xCols[-1::-1]
            except AttributeError:
                # Use all in the trajectory as a fallback...
                xkeys = self.x.cols[-1::-1]
            # Perform lexsort
            try:
                # Create a tuple of lexicon variables
                # Loop backwards through variables to prioritize first key
                XV = tuple(self[k] for k in xkeys)
                # Use lexicon sort
                I = np.lexsort(XV)
            except Exception:
                # Fall back to first key
                I = self.ArgSort(self.xCols[0])
        # Sort all fields.
        for k in self:
            # Get values
            v = self[k]
            # Skip if not a list
            if not isinstance(v, np.ndarray):
                continue
            # Skip if empty
            if len(v) == 0:
                continue
            # Sort it
            if v.ndim == 1:
                if np.max(I) < v.size:
                    self[k] = v[I]
            elif v.ndim == 2:
                if np.max(I) < v.shape[1]:
                    self[k] = v[:, I]

    # Find the index of the point in the trajectory.
    def GetRunMatrixIndex(self, j):
        r"""Find an entry in the run matrix (trajectory)

        :Call:
            >>> i = DBi.GetRunMatrixIndex(self, j)
        :Inputs:
            *DBi*: :class:`cape.cfdx.databook.DataBookComp`
                An individual item data book
            *j*: :class:`int`
                Index of the case from the databook to try match
        :Outputs:
            *i*: :class:`int`
                RunMatrix index or ``None``
        :Versions:
            * 2015-05-28 ``@ddalle``: v1.0
        """
        # Initialize indices (assume all trajectory points match to start).
        i = np.arange(self.x.nCase)
        # Loop through keys requested for matches.
        for k in self.x.cols:
            # Get the target value from the data book.
            v = self[k][j]
            # Search for matches.
            try:
                # Filter test criterion.
                ik = np.where(self.x[k] == v)[0]
                # Check if the last element should pass but doesn't.
                if (v == self.x[k][-1]):
                    # Add the last element.
                    ik = np.union1d(ik, [self.x.nCase-1])
                # Restrict to rows that match above.
                i = np.intersect1d(i, ik)
            except Exception:
                return None
        # Output
        try:
            # There should be one match.
            return i[0]
        except Exception:
            # No matches.
            return None

    # Find an entry by trajectory variables.
    def FindMatch(self, i: int) -> Optional[int]:
        r"""Find an entry by run matrix (trajectory) variables

        It is assumed that exact matches can be found. However,
        run matrix keys that do not affect the name of the folder

        :Call:
            >>> j = db.FindMatch(i)
        :Inputs:
            *db*: :class:`DataBookComp`
                An individual-component data book
            *i*: :class:`int`
                Run matrix index to match
        :Outputs:
            *j*: ``None`` | :class:`int`
                Index of databook entry that matches run matrix case *i*
        :Versions:
            * 2014-12-22 ``@ddalle``: v1.0
            * 2024-10-16 ``@ddalle``: v1.1; replace np.nan -> None
        """
        # Initialize indices (assume all are matches)
        j = np.arange(self.n) > -1
        # Loop through keys requested for matches.
        for k in self.x.cols:
            # Determine whether or not this variable affects folder name
            q = self.x.defns[k].get("Label", True)
            # If not, skip this test
            if not q:
                continue
            # Get the target value (from the trajectory)
            v = self.x[k][i]
            # Search for matches.
            try:
                # Combine criteria
                j = np.logical_and(j, self[k] == v)
            except Exception:
                # Untestable
                pass
        # Convert booleans to indices
        mask, = np.where(j)
        # There should be exactly one match
        if mask.size > 0:
            return mask[0]

    # Find run matrix index
    def FindCaseIndex(self, j: int) -> Optional[int]:
        r"""Find index of databook entry *j* in run matrix (if any)

        :Call:
            >>> i = db.FindCaseIndex(j)
        :Inputs:
            *db*: :class:`DataBookComp`
                Single databook component
            *j*: :class:`int`
                Databook index
        :Outputs:
            *i*: ``None`` | :class:`int`
                Run matrix index, if applicable
        :Versions:
            * 2024-10-16 ``@ddalle``: v1.0
        """
        # Get conditions
        v = [self[col][j] for col in self.x.cols]
        # Form name
        name = self.x.genr8_fullname(v)
        # Search for it
        return self.x.GetCaseIndex(name)

    # Find an entry using specified tolerance options
    def FindTargetMatch(self, DBT, i, topts={}, keylist='tol', **kw):
        r"""Find a target entry by run matrix (trajectory) variables

        Cases will be considered matches by comparing variables
        specified in the *topts* variable, which shares some of the
        options from the ``"Targets"`` subsection of the ``"DataBook"``
        section of ``cape.json``. Suppose that *topts* contains the
        following

        .. code-block:: python

            {
                "RunMatrix": {"alpha": "ALPHA", "Mach": "MACH"}
                "Tolerances": {
                    "alpha": 0.05,
                    "Mach": 0.01
                },
                "Keys": ["alpha", "Mach", "beta"]
            }

        Then any entry in the data book target that matches the Mach
        number within 0.01 (using a column labeled ``"MACH"``) and alpha
        to within 0.05 is considered a match.  Because the *Keys*
        parameter contains ``"beta"``, the search will also look for
        exact matches in ``"beta"``.

        If the *Keys* parameter is not set, the search will use either
        all the keys in the trajectory, *x.cols*, or just the keys
        specified in the ``"Tolerances"`` section of *topts*. Which of
        these two default lists to use is determined by the *keylist*
        input.

        :Call:
            >>> j = DBc.FindTargetMatch(DBT, i, topts, **kw)
        :Inputs:
            *DBc*: :class:`cape.cfdx.databook.DataBookComp`
                Instance of original databook
            *DBT*: :class:`DataBookComp` | :class:`TargetDataBook`
                Target databook of any type
            *i*: :class:`int`
                Case index from *DBc.x* for *DBT.x* to match
            *topts*: :class:`dict` | :class:`TargetDataBook`
                Criteria used to determine a match
            *keylist*: {``"x"``} | ``"tol"``
                Test key source: ``x.cols`` | ``topts.Tolerances``
            *source*: ``"self"`` | {``"target"``}
                Match *DBc.x* case *i* if ``"self"``, else *DBT.x* *i*
        :Outputs:
            *j*: :class:`numpy.ndarray`\ [:class:`int`]
                Array of indices that match within tolerances
        :See also:
            * :func:`cape.cfdx.databook.TargetDataBook.FindMatch`
            * :func:`cape.cfdx.databook.DataBookComp.FindMatch`
        :Versions:
            * 2014-12-21 ``@ddalle``: v1.0
            * 2016-06-27 ``@ddalle``: v1.1; Moved from TargetDataBook
            * 2018-02-12 ``@ddalle``: v1.2; First arg ``DataBookComp``
        """
        # Assign source and target
        if kw.get("source", "self").lower() in ["target", "targ"]:
            # Match self.x[j] to DBT.x[i]
            DB1 = DBT
            DB2 = self
        else:
            # Match DBT.x[j] to self.x[j]
            DB1 = self
            DB2 = DBT
        # Initialize indices (assume all are matches)
        n = len(DB2[list(DB2.keys())[0]])
        J = np.arange(n) > -1
        # Interpret trajectory options for both databooks
        try:
            topts1 = DB1.topts
        except Exception:
            topts1 = topts
        try:
            topts2 = DB2.topts
        except Exception:
            topts2 = topts
        # Get the trajectory key translations.   This determines which keys to
        # filter and what those keys are called in the source file.
        tkeys1 = topts1.get('RunMatrix', {})
        tkeys2 = topts2.get('RunMatrix', {})
        # Tolerance options
        tolopts1 = topts1.get('Tolerances', {})
        tolopts2 = topts2.get('Tolerances', {})
        # Extract trajectories
        x1 = DB1.x
        x2 = DB2.x
        # Ensure target trajectory corresponds to its contents
        DB1.UpdateRunMatrix()
        DB2.UpdateRunMatrix()
        # Get list of keys to match
        if keylist.lower() == 'x':
            # Use all trajectory keys as default
            keys = topts.get('Keys', DB1.x.cols)
        else:
            # Use the tolerance keys
            keys = topts.get('Keys', tolopts2.keys())
        # Loop through keys requested for matches.
        for k in keys:
            # Get the name of the column according to the source file.
            col1 = tkeys1.get(k, k)
            col2 = tkeys2.get(k, k)
            # Skip it if key not recognized
            if col1 is None:
                continue
            if col2 is None:
                continue
            # Get the tolerance.
            tol1 = tolopts1.get(k, 0.0)
            tol2 = tolopts2.get(k, 0.0)
            # Skip if tolerance blocked out
            if tol1 is None:
                continue
            if tol2 is None:
                continue
            # Use maximum tolerance
            tol = max(tol1, tol2)
            # Check key type (don't filter strings)
            v1 = x1.defns.get(k, {}).get("Value", "float")
            v2 = x2.defns.get(k, {}).get("Value", "float")
            # Check for string/unicode
            if v1 in ["str", "unicode"]:
                continue
            if v2 in ["str", "unicode"]:
                continue
            # Get target value
            if col1 in DB1:
                # Take value from column
                v = DB1[col1][i]
            if k in x1.cols:
                # Directly available
                v = x1[k][i]
            elif k == "alpha":
                # Get angle of attack
                v = x1.GetAlpha(i)
            elif k == "beta":
                # Get sideslip
                v = x1.GetBeta(i)
            elif k in ["alpha_t", "aoav"]:
                # Get total angle of attack
                v = x1.GetAlphaTotal(i)
            elif k in ["phi", "phiv"]:
                # Get velocity roll angle
                v = x1.GetPhi(i)
            elif k in ["alpha_m", "aoam"]:
                # Get maneuver angle of attack
                v = x1.GetAlphaManeuver(i)
            elif k in ["phi_m", "phim"]:
                # Get maneuver roll angle
                v = x1.GetPhiManeuver(i)

            # Get value
            if col2 in DB2:
                # Extract value
                V = DB2[col2]
            elif k in x2.keys:
                # Available in the trajectory
                V = x2[k]
            elif (k == "alpha"):
                # Get angle of attack
                V = x2.GetAlpha()
            elif (k == "beta"):
                # Get angle of sideslip
                V = x2.GetBeta()
            elif (k in ["alpha_t", "aoav"]):
                # Get total angle of attack
                V = x2.GetAlphaTotal()
            elif (k in ["phi", "phiv"]):
                # Get velocity roll angle
                V = x2.GetPhi()
            elif (k in ["alpha_m", "aoam"]):
                # Get maneuver angle of attack
                V = x2.GetAlphaManeuver()
            elif (k in ["phi_m", "phim"]):
                # Get maneuver roll angle
                V = x2.GetPhiManeuver()

            # Test
            qk = np.abs(v-V) <= tol
            # Check for special modifications
            if k in ["phi", "phi_m", "phiv", "phim"]:
                # Get total angle of attack
                aoav = x2.GetAlphaTotal()
                # Combine *phi* constraint with any *aoav==0* case
                qk = np.logical_or(qk, np.abs(aoav) <= 1e-10)
            # Combine constraints
            J = np.logical_and(J, qk)

        # Output
        return np.where(J)[0]

    # Find data book match
    def FindDBMatch(self, DBc, i: int):
        r"""Find the index of an exact match to case *i* in another databook

        :Call:
            >>> j = DBi.FindDBMatch(DBc, i)
        :Inputs:
            *DBi*: :class:`cape.cfdx.databook.DataBookComp`
                Data book base object
            *DBc*: :class:`cape.cfdx.databook.DataBookComp`
                Another data book base object
            *i*: :class:`int`
                Data book index for *DBi*
        :Outputs:
            *j*: ``None`` | :class:`int`
                Data book index for *DBj*
        :Versions:
            * 2017-06-26 ``@ddalle``: v1.0
        """
        # Initialize indices of potential matches
        J = np.arange(DBc.n)
        # Loop through keys
        for k in self.x.cols:
            # Determine whether or not this variable affects folder name
            q = self.x.defns[k].get("Label", True)
            # If not, skip this test
            if not q:
                continue
            # Get value
            v = self[k][i]
            # Check match
            try:
                # Find indices of matches
                jk = np.where(DBc[k][J] == v)[0]
                # Check for at least one match
                if len(jk) == 0:
                    return None
                # Restrict to rows that match above
                J = J[jk]
            except Exception:
                # Ignore this column
                pass
        # Check output
        if len(J) > 1:
            # Multiple matches
            return J
        elif len(J) == 1:
            # Single match
            return J[0]
        else:
            # No match?
            return None

    # Find an entry using specified tolerance options
    def FindCoSweep(self, x, i, EqCons=[], TolCons={}, GlobCons=[], xkeys={}):
        r"""Find data book entries meeting constraints seeded from point *i*

        Cases will be considered matches if data book values match trajectory
        *x* point *i*.  For example, if we have the following values for
        *EqCons* and *TolCons* have the following values:

        .. code-block:: python

            EqCons = ["beta"]
            TolCons = {"alpha": 0.05, "mach": 0.01}

        Then this method will compare *DBc["mach"]* to *x.mach[i]*.  Any case
        such that pass all of the following tests will be included.

        .. code-block:: python

            abs(DBc["mach"] - x.mach[i]) <= 0.01
            abs(DBc["alpha"] - x.alpha[i]) <= 0.05
            DBc["beta"] == x.beta[i]

        All entries must also meet a list of global constraints from
        *GlobCons*.  Users can also use *xkeys* as a dictionary of alternate
        key names to compare to the trajectory.  Consider the following values:

        .. code-block:: python

            TolCons = {"alpha": 0.05}
            xkeys = {"alpha": "AOA"}

        Then the test becomes:

        .. code-block:: python

            abs(DBc["AOA"] - x.alpha[i]) <= 0.05

        :Call:
            >>> J = DBc.FindCoSweep(x, i, EqCons={}, TolCons={}, **kw)
        :Inputs:
            *DBc*: :class:`cape.cfdx.databook.DataBookComp`
                Data book component instance
            *x*: :class:`cape.runmatrix.RunMatrix`
                RunMatrix (i.e. run matrix) to use for target value
            *i*: :class:`int`
                Index of the case from the trajectory to try match
            *EqCons*: {``[]``} | :class:`list` (:class:`str`)
                List of variables that must match the trajectory exactly
            *TolCons*: {``{}``} | :class:`dict`\ [:class:`float`]
                List of variables that may match trajectory within a tolerance
            *GlobCons*: {``[]``} | :class:`list` (:class:`str`)
                List of global constraints, see :func:`cape.RunMatrix.Filter`
            *xkeys*: {``{}``} | :class:`dict` (:class:`str`)
                Dictionary of alternative names of variables
        :Outputs:
            *J*: :class:`numpy.ndarray`\ [:class:`int`]
                Array of indices that match the trajectory within tolerances
        :See also:
            * :func:`cape.cfdx.databook.TargetDataBook.FindMatch`
            * :func:`cape.cfdx.databook.DataBookComp.FindMatch`
        :Versions:
            * 2014-12-21 ``@ddalle``: v1.0
            * 2016-06-27 ``@ddalle``: Moved from TargetDataBook and generalized
        """
        # Initialize indices (assume all are matches)
        n = len(self[list(self.keys())[0]])
        J = np.arange(n) > -1
        # De-None-ify
        GlobCons = [] if GlobCons is None else GlobCons
        TolCons = {} if TolCons is None else TolCons
        EqCons = [] if EqCons is None else EqCons
        xkeys = {} if xkeys is None else xkeys
        # Check types
        ti  = i.__class__.__name__
        tx  = x.__class__.__name__
        teq = EqCons.__class__.__name__
        ttc = TolCons.__class__.__name__
        tgc = GlobCons.__class__.__name__
        txk = xkeys.__class__.__name__
        # Check types
        if not ti.startswith("int"):
            raise TypeError("RunMatrix index must be integer")
        if tx != "RunMatrix":
            raise TypeError("Input must be of class 'RunMatrix'")
        if teq != "list":
            raise TypeError("Equality constraints must be list of strings")
        if ttc != "dict":
            raise TypeError("Tolerance constraints must be a dict")
        if tgc != "list":
            raise TypeError("Global constraints must be a list of strings")
        if txk != "dict":
            raise TypeError("Key translations must be dict")

        # Apply global constraints...
        for con in GlobCons:
            try:
                # Loop through trajectory keys
                for k in x.cols:
                    # Substitute if appropriate
                    if k in con:
                        con = con.replace(k, 'self["%s"]' % xkeys.get(k, k))
                # Perform equation
                J = np.logical_and(J, eval(con))
            except Exception:
                print("    Constraint '%s' failed to evaluate." % con)

        # Loop through *EqCons*
        for k in EqCons:
            # Test if key is present
            if k in x.cols:
                # Get target value
                v = x[k][i]
            elif k == "alpha":
                # Get angle of attack
                v = x.GetAlpha(i)
            elif k == "beta":
                # Get sideslip
                v = x.GetBeta(i)
            elif k in ["alpha_t", "aoav"]:
                # Get total angle of attack
                v = x.GetAlphaTotal(i)
            elif k in ["phi", "phiv"]:
                # Get velocity roll angle
                v = x.GetPhi(i)
            elif k in ["alpha_m", "aoam"]:
                # Get maneuver angle of attack
                v = x.GetAlphaManeuver(i)
            elif k in ["phi_m", "phim"]:
                # Get maneuver roll angle
                v = x.GetPhiManeuver(i)
            # Get name of column
            col = xkeys.get(k, k)
            # Get value
            if col in self:
                # Extract value
                V = self[col]
            elif (k == "alpha") or (col == "alpha"):
                # Ensure trajectory matches
                self.UpdateRunMatrix()
                # Get angle of attack
                V = self.x.GetAlpha()
            elif (k == "beta") or (col == "beta"):
                # Get angle of sideslip
                self.UpdateRunMatrix()
                V = self.x.GetBeta()
            elif (k in ["alpha_t", "aoav"]) or (col in ["alpha_t", "aoav"]):
                # Get maneuver angle of attack
                self.UpdateRunMatrix()
                V = self.x.GetAlphaTotal()
            elif (k in ["phi", "phiv"]) or (col in ["phi", "phiv"]):
                # Get maneuver roll angle
                self.UpdateRunMatrix()
                V = self.x.GetPhi()
            elif (k in ["alpha_m", "aoam"]) or (col in ["alpha_m", "aoam"]):
                # Get maneuver angle of attack
                self.UpdateRunMatrix()
                V = self.x.GetAlphaManeuver()
            elif (k in ["phi_m", "phim"]) or (col in ["phi_m", "phim"]):
                # Get maneuver roll angle
                self.UpdateRunMatrix()
                V = self.x.GetPhiManeuver()
            # Evaluate constraint
            qk = np.abs(v - V) <= 1e-10
            # Check for special modifications
            if k in ["phi", "phi_m", "phiv", "phim"]:
                # Get total angle of attack
                aoav = self.x.GetAlphaTotal()
                # Combine *phi* constraint with any *aoav==0* case
                qk = np.logical_or(qk, np.abs(aoav) <= 1e-10)
            # Combine constraints
            J = np.logical_and(J, qk)
        # Loop through *TolCons*
        for k in TolCons:
            # Test if key is present
            if k in x.cols:
                # Get target value
                v = x[k][i]
            elif k == "alpha":
                # Get angle of attack
                v = x.GetAlpha(i)
            elif k == "beta":
                # Get sideslip
                v = x.GetBeta(i)
            elif k in ["alpha_t", "aoav"]:
                # Get total angle of attack
                v = x.GetAlphaTotal(i)
            elif k in ["phi", "phiv"]:
                # Get velocity roll angle
                v = x.GetPhi(i)
            elif k in ["alpha_m", "aoam"]:
                # Get maneuver angle of attack
                v = x.GetAlphaManeuver(i)
            elif k in ["phi_m", "phim"]:
                # Get maneuver roll angle
                v = x.GetPhiManeuver(i)
            # Get name of column
            col = xkeys.get(k, k)
            # Get value
            if col in self:
                # Extract value
                V = self[col]
            elif (k == "alpha") or (col == "alpha"):
                # Ensure trajectory matches
                self.UpdateRunMatrix()
                # Get angle of attack
                V = self.x.GetAlpha()
            elif (k == "beta") or (col == "beta"):
                # Get angle of sideslip
                self.UpdateRunMatrix()
                V = self.x.GetBeta()
            elif (k in ["alpha_t", "aoav"]) or (col in ["alpha_t", "aoav"]):
                # Get maneuver angle of attack
                self.UpdateRunMatrix()
                V = self.x.GetAlphaTotal()
            elif (k in ["phi", "phiv"]) or (col in ["phi", "phiv"]):
                # Get maneuver roll angle
                self.UpdateRunMatrix()
                V = self.x.GetPhi()
            elif (k in ["alpha_m", "aoam"]) or (col in ["alpha_m", "aoam"]):
                # Get maneuver angle of attack
                self.UpdateRunMatrix()
                V = self.x.GetAlphaManeuver()
            elif (k in ["phi_m", "phim"]) or (col in ["phi_m", "phim"]):
                # Get maneuver roll angle
                self.UpdateRunMatrix()
                V = self.x.GetPhiManeuver()
            # Get tolerance
            tol = TolCons[k]
            # Test
            qk = np.abs(v-V) <= tol
            # Check for special modifications
            if k in ["phi", "phi_m", "phiv", "phim"]:
                # Get total angle of attack
                self.UpdateRunMatrix()
                aoav = self.x.GetAlphaTotal()
                # Combine *phi* constraint with any *aoav==0* case
                qk = np.logical_or(qk, np.abs(aoav) <= 1e-10)
            # Combine constraints
            J = np.logical_and(J, qk)
        # Output (convert boolean array to indices)
        return np.where(J)[0]
  # >

  # =============
  # Statistics
  # =============
  # <
   # ----------
   # Deltas
   # ----------
   # [
    # Get statistics on deltas for a subset
    def GetDeltaStats(self, DBT, comp, coeff, I, topts={}, **kw):
        r"""Calculate statistics on differences between two databooks

        :Call:
            >>> S = DBc.GetDeltaStats(DBT, coeff, I, topts=None, **kw)
        :Inputs:
            *DBc*: :class:`cape.cfdx.databook.DataBookComp`
                Component databook
            *coeff*: :class:`str`
                Name of coefficient on which to compute statistics
            *I*: :class:`list`\ [:class:`int`]
                Indices of cases/entries to consider
            *topts*: {``{}``} | :class:`dict`
                Dictionary of tolerances for variables in question
            *keylist*: {``"x"``} | ``"tol"``
                Default test key source: ``x.cols`` or ``topts.Tolerances``
            *CombineTarget*: {``True``} | ``False``
                For cases with multiple matches, compare to mean target value
        :Outputs:
            *S*: :class:`dict`
                Dictionary of statistical results
            *S["delta"]*: :class:`np.ndarray`
                Array of deltas for each valid case
            *S["n"]*: :class:`int`
                Number
            *S["mu"]*: :class:`float`
                Mean of histogram
        :Versions:
            * 2018-02-12 ``@ddalle``: v1.0
        """
        # Process mean target option
        qmu = kw.get("CombineTarget", True)
        # Initialize target indices
        JT = []
        # Initialize deltas
        D = []
        # Initialize count
        n = 0
        # Loop through cases
        for i in I:
            # Find target
            Ji = self.FindTargetMatch(DBT, i, topts, keylist="tol")
            # Get the value
            v = self.GetCoeff(comp, coeff, i)
            # Number of matches
            ni = len(Ji)
            # Count
            n += ni
            # Save entries
            for j in Ji:
                JT.append([i, j])
            # Process targets
            if qmu:
                # Check number of matches
                if ni == 0:
                    # Save a NaN
                    D.append(np.nan)
                elif ni == 1:
                    # Get the target value
                    w = DBT.GetCoeff(comp, coeff, j)
                    # Save the value
                    D.append(v - w)
                else:
                    # Get the target values
                    w = np.mean(DBT.GetCoeff(comp, coeff, Ji))
                    # Save the delta
                    D.append(v - w)
            else:
                # Check number of matches
                if ni == 0:
                    # No match
                    continue
                else:
                    # Get all values
                    W = DBT.GetCoeff(comp, coeff, Ji)
                    # Append each
                    for w in W:
                        D.append(v - w)
        # Create array
        D = np.array(D)
        JT = np.array(JT)
        # Form output
        return {
            "mu": np.mean(D),
            "delta": D,
            "index": JT,
            "n": n
        }
   # ]
  # >

  # =====
  # Plot
  # =====
  # <
    # Plot a sweep of one or more coefficients
    def PlotCoeffBase(self, coeff, I, **kw):
        r"""Plot sweep of one coefficient or quantity over several cases

        This is the base method upon which data book sweep plotting is
        built. Other methods may call this one with modifications to
        the default settings. For example
        :func:`cape.cfdx.databook.TargetDataBook.PlotCoeff` changes the
        default *PlotOptions* to show a red line instead of the standard
        black line.  All settings can still be overruled by explicit
        inputs to either this function or any of its children.

        :Call:
            >>> h = DBi.PlotCoeffBase(coeff, I, **kw)
        :Inputs:
            *DBi*: :class:`cape.cfdx.databook.DataBookComp`
                An individual item data book
            *coeff*: :class:`str`
                Coefficient being plotted
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indexes of cases to include in sweep
        :Keyword Arguments:
            *x*: {``None``} | :class:`str`
                RunMatrix key for *x* axis (or plot against index if ``None``)
            *Label*: {*comp*} | :class:`str`
                Manually specified label
            *Legend*: {``True``} | ``False``
                Whether or not to use a legend
            *StDev*: {``None``} | :class:`float`
                Multiple of iterative history standard deviation to plot
            *MinMax*: {``False``} | ``True``
                Whether to plot minimum and maximum over iterative history
            *Uncertainty*: {``False``} | ``True``
                Whether to plot direct uncertainty
            *PlotOptions*: :class:`dict`
                Plot options for the primary line(s)
            *StDevOptions*: :class:`dict`
                Dictionary of plot options for the standard deviation plot
            *MinMaxOptions*: :class:`dict`
                Dictionary of plot options for the min/max plot
            *UncertaintyOptions*: :class:`dict`
                Dictionary of plot options for the uncertainty plot
            *FigureWidth*: :class:`float`
                Width of figure in inches
            *FigureHeight*: :class:`float`
                Height of figure in inches
            *PlotTypeStDev*: {``'FillBetween'``} | ``'ErrorBar'``
                Plot function to use for standard deviation plot
            *PlotTypeMinMax*: {``'FillBetween'``} | ``'ErrorBar'``
                Plot function to use for min/max plot
            *PlotTypeUncertainty*: ``'FillBetween'`` | {``'ErrorBar'``}
                Plot function to use for uncertainty plot
            *LegendFontSize*: {``9``} | :class:`int` > 0 | :class:`float`
                Font size for use in legends
            *Grid*: {``None``} | ``True`` | ``False``
                Turn on/off major grid lines, or leave as is if ``None``
            *GridStyle*: {``{}``} | :class:`dict`
                Dictionary of major grid line line style options
            *MinorGrid*: {``None``} | ``True`` | ``False``
                Turn on/off minor grid lines, or leave as is if ``None``
            *MinorGridStyle*: {``{}``} | :class:`dict`
                Dictionary of minor grid line line style options
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: v1.0
            * 2015-12-14 ``@ddalle``: Added error bars
        """
       # ----------------
       # Initial Options
       # ----------------
        # Make sure the plotting modules are present.
        plt = import_pyplot()
        # Get horizontal key.
        xk = kw.get('x')
        # Figure dimensions
        fw = kw.get('FigureWidth', 6)
        fh = kw.get('FigureHeight', 4.5)
        # Iterative uncertainty options
        qmmx = kw.get('MinMax', False)
        qerr = kw.get('Uncertainty', False)
        ksig = kw.get('StDev')
        # Get plot types
        tmmx = kw.get('PlotTypeMinMax', 'FillBetween')
        terr = kw.get('PlotTypeUncertainty', 'ErrorBar')
        tsig = kw.get('PlotTypeStDev', 'FillBetween')
        # Initialize output
        h = {}
        # Process component name
        try:
            # Default to attribute (pyCart dataBook components store comp)
            comp = kw.get('comp', self.comp)
        except AttributeError:
            # Generic targets do not have a dedicated "component"
            comp = kw.get('comp')
       # ---------------------
       # Reference Parameters
       # ---------------------
        # Get reference quantities
        Lref = self.opts.get_RefLength(comp)
        MRP  = self.opts.get_RefPoint(comp)
        # Unpack MRP
        if MRP is None:
            # None
            xMRP = 0.0
            yMRP = 0.0
            zMRP = 0.0
        else:
            # Unpack
            xMRP, yMRP, zMRP = MRP
       # --------------------
       # Process x-axis data
       # --------------------
        # Extract the values for the x-axis.
        if xk is None or xk == 'Index':
            # Use the indices as the x-axis
            xv = I
            # Label
            xk = 'Index'
        elif xk in self:
            # Extract the values.
            xv = self[xk][I]
        elif xk == "alpha":
            # Update trajectory
            self.UpdateRunMatrix()
            # Get angles of attack
            xv = self.x.GetAlpha(I)
        elif xk == "beta":
            # Update trajectory
            self.UpdateRunMatrix()
            # Get sideslip angles
            xv = self.x.GetBeta(I)
        elif xk in ["alpha_t", "aoav"]:
            # Update trajectory
            self.UpdateRunMatrix()
            # Get maneuver angle of attack
            xv = self.x.GetAlphaTotal(I)
        elif xk in ["phi", "phiv"]:
            # Update trajectory
            self.UpdateRunMatrix()
            # Get maneuver roll angles
            xv = self.x.GetPhi(I)
        elif xk in ["alpha_m", "aoam"]:
            # Update trajectory
            self.UpdateRunMatrix()
            # Get maneuver angle of attack
            xv = self.x.GetAlphaManeuver(I)
        elif xk in ["phi_m", "phim"]:
            # Update trajectory
            self.UpdateRunMatrix()
            # Get maneuver roll angles
            xv = self.x.GetPhiManeuver(I)
        # Sorting order for *xv*
        ixv = np.argsort(xv)
        xv = xv[ixv]
       # --------------------
       # Process y-axis data
       # --------------------
        # Extract the mean values.
        if coeff in self:
            # Read the coefficient directly
            yv = self[coeff][I]
        elif coeff in ["CF", "CT"]:
            # Try getting magnitude of force
            yv = np.sqrt(
                self["CA"][I]**2 +
                self["CY"][I]**2 + self["CN"][I]**2)
        elif coeff in ["CL"]:
            # Get angle of attack
            alph = self.x.GetAlpha(I)
            # Convert alpha to radians
            alph *= DEG
            # Try calculating lift coefficient
            yv = self["CN"][I]*np.cos(alph) - self["CA"][I]*np.sin(alph)
        elif coeff in ["CD"]:
            # Get angle of attack
            alph = self.x.GetAlpha(I)
            # Get angle of sideslip
            beta = self.x.GetBeta(I)
            # Check if beta returned
            if beta is None:
                # Set beta to 0.0
                beta = 0.0
            # Convert alpha to radians
            alph *= DEG
            # Convert beta to radians
            beta *= DEG
            # Try calculating drag coefficient
            yv = (
                self["CA"][I]*np.cos(alph)*np.cos(beta) +
                self["CN"][I]*np.sin(alph)*np.cos(beta))
        elif coeff in ["CP"]:
            # Try calculating center of pressure
            yv = xMRP - self["CLM"][I]*Lref/self["CN"][I]
        elif coeff in ["cp"]:
            # Try calculating center of pressure (nondimensional)
            yv = xMRP/Lref - self["CLM"][I]/self["CN"][I]
        elif coeff in ["CPY"]:
            # Try calculating center of pressure
            yv = xMRP - self["CLN"][I]*Lref/self["CY"][I]
        elif coeff in ["cpy"]:
            # Try calculating center of pressure (nondimensional)
            yv = xMRP/Lref - self["CLN"][I]/self["CY"][I]
        else:
            raise ValueError("Unrecognized coefficient '%s'" % coeff)
        # Check for override parameters
        Lref = kw.get("Lref", Lref)
        # Check for special cases
        if coeff == "CLM":
            # Check for MRP shift
            xmrp  = kw.get("XMRP")
            dxmrp = kw.get("DXMRP")
            fxmrp = kw.get("XMRPFunction")
            # Shift if necessary
            if (xmrp is not None) and ("CN" in self):
                # Check type
                if xmrp.__class__.__name__ == "list":
                    xmrp = np.array(xmrp)
                # Shift moment to specific point
                yv = yv + (xmrp-xMRP)/Lref*self["CN"][I]
            if (dxmrp is not None) and ("CN" in self):
                # Check type
                if dxmrp.__class__.__name__ == "list":
                    dxmrp = np.array(dxmrp)
                # Shift the moment reference point
                yv = yv + dxmrp/Lref*self["CN"][I]
            if (fxmrp is not None) and ("CN" in self):
                # Use a function to evaluate new MRP (may vary by index)
                xmrp = fxmrp(self, I)
                # Shift the moment to specific point
                yv = yv + (xmrp-xMRP)/Lref*self["CN"][I]
        elif coeff == "CLN":
            # Check for MRP shift
            xmrp  = kw.get("XMRP")
            dxmrp = kw.get("DXMRP")
            fxmrp = kw.get("XMRPFunction")
            # Shift if necessary
            if (xmrp is not None) and ("CY" in self):
                # Shift moment to specific point
                yv = yv + (xmrp-xMRP)/Lref*self["CY"][I]
            if (dxmrp is not None) and ("CY" in self):
                # Shift the moment reference point
                yv = yv + dxmrp/Lref*self["CY"][I]
            if (fxmrp is not None) and ("CY" in self):
                # Use a function to evaluate new MRP (may vary by index)
                xmrp = fxmrp(self, I)
                # Shift the moment to specific point
                yv = yv + (xmrp-xMRP)/Lref*self["CY"][I]
        # Sort the data
        yv = yv[ixv]
        # Default label starter
        try:
            # Name of component
            try:
                # In some cases *name* is more specific than
                dlbl = self.name
            except AttributeError:
                # Component is usually there
                dlbl = self.comp
        except AttributeError:
            # Backup default
            try:
                # Name of object
                dlbl = self.Name
            except AttributeError:
                # No default
                dlbl = ''
        # Initialize label.
        lbl = kw.get('Label', dlbl)
       # -----------------------
       # Standard Deviation Plot
       # -----------------------
        # Standard deviation fields
        cstd = coeff + "_std"
        # Show iterative standard deviation.
        if ksig and (cstd in self):
            # Initialize plot options for standard deviation
            if tsig == "ErrorBar":
                # Error bars
                kw_s = DBPlotOpts(color='b', fmt=None, zorder=1)
            else:
                # Filled region
                kw_s = DBPlotOpts(
                    color='b', lw=0.0,
                    facecolor='b', alpha=0.35, zorder=1)
            # Add standard deviation to label.
            lbl = u'%s (\u00B1%s\u03C3)' % (lbl, ksig)
            # Extract plot options from keyword arguments.
            for k in util.denone(kw.get("StDevOptions")):
                # Option.
                o_k = kw["StDevOptions"][k]
                # Override the default option.
                if o_k is not None:
                    kw_s[k] = o_k
            # Get the standard deviation value.
            sv = self[cstd][I][ixv]
            # Check plot type
            if tsig == "ErrorBar":
                # Error bars
                h['std'] = plt.errorbar(xv, yv, yerr=ksig*sv, **kw_s)
            else:
                # Filled region
                h['std'] = plt.fill_between(xv, yv-ksig*sv, yv+ksig*sv, **kw_s)
       # ------------
       # Min/Max Plot
       # ------------
        # Min/max fields
        cmin = coeff + "_min"
        cmax = coeff + "_max"
        # Show min/max options
        if qmmx and (cmin in self) and (cmax in self):
            # Initialize plot options for min/max
            if tmmx == "ErrorBar":
                # Default error bar options
                kw_m = DBPlotOpts(color='g', fmt=None, zorder=2)
            else:
                # Default filled region options
                kw_m = DBPlotOpts(
                    color='g', lw=0.0,
                    facecolor='g', alpha=0.35, zorder=2)
            # Add min/max to label.
            lbl = u'%s (min/max)' % (lbl)
            # Extract plot options from keyword arguments.
            for k in util.denone(kw.get("MinMaxOptions")):
                # Option
                o_k = kw["MinMaxOptions"][k]
                # Override the default option.
                if o_k is not None:
                    kw_m[k] = o_k
            # Get the min and max values.
            ymin = self[cmin][I]
            ymax = self[cmax][I]
            # Plot it.
            if tmmx == "ErrorBar":
                # Form +\- error bounds
                yerr = np.vstack((yv-ymin, ymax-yv))
                # Plot error bars
                h['max'] = plt.errorbar(xv, yv, yerr=yerr, **kw_m)
            else:
                # Filled region
                h['max'] = plt.fill_between(xv, ymin, ymax, **kw_m)
       # ----------------
       # Uncertainty Plot
       # ----------------
        # Uncertainty databook files
        cu = coeff + "_u"
        cuP = coeff + "_uP"
        cuM = coeff + "_uM"
        # Show uncertainty option
        if qerr and (cu in self) or (cuP in self and cuM in self):
            # Initialize plot options for uncertainty
            if terr == "FillBetween":
                # Default filled region options
                kw_u = DBPlotOpts(
                    color='c', lw=0.0,
                    facecolor='c', alpha=0.35, zorder=3)
            else:
                # Default error bar options
                kw_u = DBPlotOpts(color='c', fmt=None, zorder=3)
            # Add uncertainty to label
            lbl = u'%s UQ bounds' % (lbl)
            # Extract plot options from keyword arguments.
            for k in util.denone(kw.get("UncertaintyOptions")):
                # Option
                o_k = kw["UncertaintyOptions"][k]
                # Override the default option.
                if o_k is not None:
                    kw_u[k] = o_k
            # Get the uncertainty values.
            if cuP in self:
                # Plus and minus coefficients are given
                yuP = self[cuP]
                yuM = self[cuM]
            else:
                # Single uncertainty
                yuP = self[cu]
                yuM = yuP
            # Plot
            if terr == "FillBetween":
                # Form min and max
                ymin = yv - yuM
                ymax = yv + yuP
                # Filled region
                h['err'] = plt.fill_between(xv, ymin, ymax, **kw_u)
            else:
                # Form +/- error bounds
                yerr = np.vstack((yuM, yuP))
                # Plot error bars
                h['err'] = plt.fill_between(xv, yv, yerr, **kw_u)
       # ------------
       # Primary Plot
       # ------------
        # Initialize plot options for primary plot
        kw_p = DBPlotOpts(color='k', marker='^', zorder=9, linestyle='-')
        # Plot options
        for k in util.denone(kw.get("PlotOptions")):
            # Option
            o_k = kw["PlotOptions"][k]
            # Override the default option.
            if o_k is not None:
                kw_p[k] = o_k
        # Label
        kw_p.setdefault('label', lbl)
        # Plot it.
        h['line'] = plt.plot(xv, yv, **kw_p)
       # -------
       # Labels
       # -------
        # Get the figure and axes.
        h['fig'] = plt.gcf()
        h['ax'] = plt.gca()
        # Check for an existing ylabel
        ly = h['ax'].get_ylabel()
        # Compare to requested ylabel
        if ly and ly != coeff:
            # Combine labels.
            ly = ly + '/' + coeff
        else:
            # Use the coefficient.
            ly = coeff
        # Labels.
        h['x'] = plt.xlabel(xk)
        h['y'] = plt.ylabel(ly)
        # Get limits that include all data (and not extra).
        xmin, xmax = get_xlim(h['ax'], pad=0.05)
        ymin, ymax = get_ylim(h['ax'], pad=0.05)
        # Set defaults
        if "XMin" in kw and kw["XMin"] is None:
            kw["XMin"] = xmin
        if "XMax" in kw and kw["XMax"] is None:
            kw["XMax"] = xmax
        if "YMin" in kw and kw["YMin"] is None:
            kw["YMin"] = ymin
        if "YMax" in kw and kw["YMax"] is None:
            kw["YMax"] = ymax
        # Check for keyword arguments
        xmax = kw.get("XMax", xmax)
        xmin = kw.get("XMin", xmin)
        ymax = kw.get("YMax", ymax)
        ymin = kw.get("YMin", ymin)
        # Make sure data is included.
        h['ax'].set_xlim(xmin, xmax)
        h['ax'].set_ylim(ymin, ymax)
       # -------
       # Legend
       # -------
        # Check for legend option
        if kw.get('Legend', True):
            # Add extra room for the legend.
            h['ax'].set_ylim((ymin, 1.2*ymax-0.2*ymin))
            # Font size checks.
            if len(h['ax'].get_lines()) > 5:
                # Very small
                fsize0 = 7
            else:
                # Just small
                fsize0 = 9
            # Check for input
            fsize = kw.get("LegendFontSize", fsize0)
            # Check for "LegendFontSize=None"
            if not fsize:
                fsize = fsize0
            # Activate the legend
            _set_font()
            h['legend'] = h['ax'].legend(
                loc='upper center',
                prop=dict(size=fsize, family=FONT_FAMILY),
                bbox_to_anchor=(0.5, 1.05), labelspacing=0.5)
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
       # ---------------
       # Sizing/Margins
       # ---------------
        # Figure dimensions.
        if fh:
            h['fig'].set_figheight(fh)
        if fw:
            h['fig'].set_figwidth(fw)
        # Attempt to apply tight axes
        _tight_layout()
        # Output
        return h

    # Plot a sweep of one or more coefficients
    def PlotCoeff(self, coeff, I, **kw):
        r"""Plot a sweep of one coefficient over several cases

        :Call:
            >>> h = DBi.PlotCoeff(coeff, I, **kw)
        :Inputs:
            *DBi*: :class:`cape.cfdx.databook.DataBookComp`
                An individual item data book
            *coeff*: :class:`str`
                Coefficient being plotted
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indexes of cases to include in sweep
        :Keyword Arguments:
            * See :func:`cape.cfdx.databook.DataBookComp.PlotCoeffBase`
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: v1.0
            * 2015-12-14 ``@ddalle``: Added error bars
        """
        # Call base function with no modifications to defaults
        return self.PlotCoeffBase(coeff, I, **kw)

    # Plot a sweep of one or more coefficients
    def PlotContourBase(self, coeff, I, **kw):
        r"""Create a contour plot of selected data points

        :Call:
            >>> h = DBi.PlotContourBase(coeff, I, **kw)
        :Inputs:
            *DBi*: :class:`cape.cfdx.databook.DataBookComp`
                An individual item data book
            *coeff*: :class:`str`
                Coefficient being plotted
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indexes of cases to include in sweep
        :Keyword Arguments:
            *x*: :class:`str`
                RunMatrix key for *x* axis
            *y*: :class:`str`
                RunMatrix key for *y* axis
            *ContourType*: {"tricontourf"} | "tricontour" | "tripcolor"
                Contour plotting function to use
            *LineType*: {"plot"} | "triplot" | "none"
                Line plotting function to highlight data points
            *Label*: [ {*comp*} | :class:`str` ]
                Manually specified label
            *ColorMap*: {``"jet"``} | :class:`str`
                Name of color map to use
            *ColorBar*: [ {``True``} | ``False`` ]
                Whether or not to use a color bar
            *ContourOptions*: :class:`dict`
                Plot options to pass to contour plotting function
            *PlotOptions*: :class:`dict`
                Plot options for the line plot
            *FigureWidth*: :class:`float`
                Width of figure in inches
            *FigureHeight*: :class:`float`
                Height of figure in inches
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2017-04-17 ``@ddalle``: v1.0
        """
       # ------
       # Inputs
       # ------
        # Make sure the plotting modules are present.
        plt = import_pyplot()
        # Get horizontal key.
        xk = kw.get('x')
        yk = kw.get('y')
        # Check for axis variables
        if xk is None:
            raise ValueError("No x-axis key given")
        if yk is None:
            raise ValueError("No y-axis key given")
        # Extract the values for the x-axis
        if xk in self:
            # Get values directly
            xv = self[xk][I]
        elif xk.lower() == "alpha":
            # Angle of attack
            self.UpdateRunMatrix()
            xv = self.x.GetAlpha(I)
        elif xk.lower() == "beta":
            # Angle of sideslip
            self.UpdateRunMatrix()
            xv = self.x.GetBeta(I)
        elif xk.lower() in ["alpha_m", "aoam"]:
            # Maneuver angle of attack
            self.UpdateRunMatrix()
            xv = self.x.GetAlphaManeuver(I)
        # Extract the values for the y-axis
        if yk in self:
            # Get values directly
            yv = self[yk][I]
        elif yk.lower() == "alpha":
            # Angle of attack
            self.UpdateRunMatrix()
            yv = self.x.GetAlpha(I)
        elif yk.lower() == "beta":
            # Angle of sideslip
            self.UpdateRunMatrix()
            yv = self.x.GetBeta(I)
        elif yk.lower() in ["alpha_m", "aoam"]:
            # Maneuver angle of attack
            self.UpdateRunMatrix()
            yv = self.x.GetAlphaManeuver(I)
        # Extract the values to plot
        zv = self[coeff][I]
        # Contour type, line type
        ctyp = kw.get("ContourType", "tricontourf")
        ltyp = kw.get("LineType", "plot")
        # Convert to lower case
        if type(ctyp).__name__ in ['str', 'unicode']:
            ctyp = ctyp.lower()
        if type(ltyp).__name__ in ['str', 'unicode']:
            ltyp = ltyp.lower()
        # Figure dimensions
        fw = kw.get('FigureWidth', 6)
        fh = kw.get('FigureHeight', 4.5)
        # Initialize output
        h = {}
        # Default label starter
        try:
            # Name of component
            dlbl = self.comp
        except AttributeError:
            # Backup default
            try:
                # Name of object
                dlbl = self.Name
            except AttributeError:
                # No default
                dlbl = ''
        # Initialize label.
        lbl = kw.get('Label', dlbl)
       # ------------
       # Contour Plot
       # ------------
        # Get colormap
        ocmap = kw.get("ColorMap", "jet")
        # Initialize plot options for contour plot
        kw_c = DBPlotOpts(cmap=ocmap)
        # Controu options
        for k in util.denone(kw.get("ContourOptions")):
            # Option
            o_k = kw["ContourOptions"][k]
            # Override
            if o_k is not None:
                kw_c[k] = o_k
        # Label
        kw_c.setdefault('label', lbl)
        # Fix aspect ratio...
        if kw.get("AxisEqual", True):
            plt.axis('equal')
        # Check plot type
        if ctyp == "tricontourf":
            # Filled contour
            h['contour'] = plt.tricontourf(xv, yv, zv, **kw_c)
        elif ctyp == "tricontour":
            # Contour lines
            h['contour'] = plt.tricontour(xv, yv, zv, **kw_c)
        elif ctyp == "tripcolor":
            # Triangulation
            h['contour'] = plt.tripcolor(xv, yv, zv, **kw_c)
        else:
            # Unrecognized
            raise ValueError("Unrecognized ContourType '%s'" % ctyp)
       # ----------------
       # Line or Dot Plot
       # ----------------
        # Check for a line plot
        if ltyp and ltyp != "none":
            # Initialize plot options for primary plot
            kw_p = DBPlotOpts(color='k', marker='^', zorder=9)
            # Set default line style
            if ltyp == "plot":
                kw_p["linestyle"] = ''
            # Plot options
            for k in util.denone(kw.get("PlotOptions")):
                # Option
                o_k = kw["PlotOptions"][k]
                # Override the default option.
                if o_k is not None:
                    kw_p[k] = o_k
            # Label
            kw_p.setdefault('label', lbl)
            # Plot it
            if ltyp in ["plot", "line", "dot"]:
                # Regular plot
                h['line'] = plt.plot(xv, yv, **kw_p)
            elif ltyp == "triplot":
                # Plot triangles
                h['line'] = plt.triplot(xv, yv, **kw_p)
            else:
                # Unrecognized
                raise ValueError("Unrecognized LineType '%s'" % ltyp)
       # ----------
       # Formatting
       # ----------
        # Get the figure and axes.
        h['fig'] = plt.gcf()
        h['ax'] = plt.gca()
        # Labels.
        h['x'] = plt.xlabel(xk)
        h['y'] = plt.ylabel(yk)
        # Get limits that include all data (and not extra).
        xmin, xmax = get_xlim(h['ax'], pad=0.05)
        ymin, ymax = get_ylim(h['ax'], pad=0.05)
        # Make sure data is included.
        h['ax'].set_xlim(xmin, xmax)
        h['ax'].set_ylim(ymin, ymax)
        # Legend.
        if kw.get('ColorBar', True):
            # Font size checks.
            fsize = 9
            # Activate the color bar
            h['cbar'] = plt.colorbar()
            # Set font size
            h['cbar'].ax.tick_params(labelsize=fsize)
        # Figure dimensions.
        if fh:
            h['fig'].set_figheight(fh)
        if fw:
            h['fig'].set_figwidth(fw)
        # Attempt to apply tight axes
        _tight_layout()
        # Output
        return h

    # Plot a sweep of one or more coefficients
    def PlotContour(self, coeff, I, **kw):
        r"""Create a contour plot for a subset of cases

        :Call:
            >>> h = DBi.PlotContour(coeff, I, **kw)
        :Inputs:
            *DBi*: :class:`cape.cfdx.databook.DataBookComp`
                An individual item data book
            *coeff*: :class:`str`
                Coefficient being plotted
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indexes of cases to include in sweep
        :Keyword Arguments:
            * See :func:`cape.cfdx.databook.DataBookComp.PlotCoeffBase`
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2017-04-17 ``@ddalle``: v1.0
        """
        # Call base function with no modifications to defaults
        return self.PlotContourBase(coeff, I, **kw)

    # Plot a sweep of one or more coefficients
    def PlotHistBase(self, coeff, I, **kw):
        r"""Plot a histogram of one coefficient over several cases

        :Call:
            >>> h = DBi.PlotHistBase(coeff, I, **kw)
        :Inputs:
            *DBi*: :class:`cape.cfdx.databook.DataBookComp`
                An individual item data book
            *coeff*: :class:`str`
                Coefficient being plotted
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indexes of cases to include in sweep
        :Keyword Arguments:
            *FigureWidth*: :class:`float`
                Figure width
            *FigureHeight*: :class:`float`
                Figure height
            *Label*: [ {*comp*} | :class:`str` ]
                Manually specified label
            *Target*: {``None``} | :class:`DataBookComp` | :class:`list`
                Target database or list thereof
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
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: v1.0
            * 2015-12-14 ``@ddalle``: Added error bars
            * 2016-04-04 ``@ddalle``: Moved from point sensor to data book
        """
       # -----------
       # Preparation
       # -----------
        # Make sure the plotting modules are present.
        plt = import_pyplot()
        # Figure dimensions
        fw = kw.get('FigureWidth', 6)
        fh = kw.get('FigureHeight', 4.5)
       # -----------------
       # Statistics/Values
       # -----------------
        # Check for a target
        DBT = kw.get("target", kw.get("DBT", kw.get("Target")))
        # Figure out the "component" name
        compo = self.comp
        # Check for a point or group
        pt = getattr(self, "name", getattr(self, "pt", None))
        # Form total component name
        if pt:
            # Add component and point name
            comp = "%s.%s" % (compo, pt)
        else:
            # Just a component (like STACK_No_Base)
            comp = compo
        # Extract the values or statistics, as appropriate
        if DBT is None:
            # Extract the values
            V = self[coeff][I]
        elif type(DBT).__name__ in ["list", "ndarray"]:
            # Loop through lists until found
            for DBTc in DBT:
                # Attempt to calculate statistics
                S = self.GetDeltaStats(DBTc, comp, coeff, I)
                # Extract just the deltas
                V = S["delta"]
                V = V[np.logical_not(np.isnan(V))]
                # Exit as soon as we find at least one matching case
                if len(V) > 0:
                    break
                # (If no targets match, just use the last)
        else:
            # Get statistics
            S = self.GetDeltaStats(DBT, comp, coeff, I)
            # Extract just the deltas
            V = S["delta"]
            V = V[np.logical_not(np.isnan(V))]
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
       # ------------
       # More Options
       # ------------
        # Uncertainty options
        ksig = kw.get('StDev')
        # Reference delta
        dc = kw.get('Delta', 0.0)
        # Target values and labels
        vtarg = kw.get('TargetValue')
        ltarg = kw.get('TargetLabel')
        # Convert target values to list
        if vtarg in [None, False]:
            vtarg = []
        elif type(vtarg).__name__ not in ['list', 'tuple', 'ndarray']:
            vtarg = [vtarg]
        # Create appropriate target list for
        if type(ltarg).__name__ not in ['list', 'tuple', 'ndarray']:
            ltarg = [ltarg]
       # --------
       # Plotting
       # --------
        # Initialize dictionary of handles.
        h = {}
       # --------------
       # Histogram Plot
       # --------------
        # Initialize plot options for histogram.
        kw_h = DBPlotOpts(facecolor='c', zorder=2, bins=20)
        # Apply *Label* option if present
        lbl = kw.get("Label")
        if lbl:
            kw_h["label"] = lbl
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
            if type(r).__name__ in ['ndarray', 'list', 'tuple']:
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
       # ------------
       # Axes Handles
       # ------------
        # Get the figure and axes.
        h['fig'] = plt.gcf()
        h['ax'] = plt.gca()
        ax = h['ax']
        # Determine whether or not the distribution is normed
        q_normed = kw_h.get("normed", kw_h.get("density", False))
        # Determine whether or not the bars are vertical
        q_vert = kw_h.get("orientation", "vertical") == "vertical"
        # Get current axis limits
        if q_vert:
            xmin, xmax = ax.get_xlim()
            pmin, pmax = ax.get_ylim()
        else:
            xmin, xmax = ax.get_ylim()
            pmin, pmax = ax.get_xlim()
       # -------------
       # Gaussian Plot
       # -------------
        # Initialize options for guassian plot
        kw_g = DBPlotOpts(color='navy', lw=1.5, zorder=7)
        kw_g["label"] = "Normal distribution"
        # Extract options from kwargs
        for k in util.denone(kw.get("GaussianOptions", {})):
            # Override the default option.
            if kw["GaussianOptions"][k] is not None:
                kw_g[k] = kw["GaussianOptions"][k]
        # Check whether or not to plot it
        if q_normed and kw.get("PlotGaussian"):
            # Lookup probabilities
            xval = np.linspace(xmin, xmax, 151)
            # Compute Gaussian distribution
            yval = 1/(vstd*np.sqrt(2*np.pi))*np.exp(-0.5*((xval-vmu)/vstd)**2)
            # Check orientation
            if q_vert:
                # Plot a vertical line for the mean.
                h['mean'] = plt.plot(xval, yval, **kw_g)
            else:
                # Plot a horizontal line for th emean.
                h['mean'] = plt.plot(yval, xval, **kw_g)
       # ---------
       # Mean Plot
       # ---------
        # Initialize options for mean plot
        kw_m = DBPlotOpts(color='k', lw=2, zorder=6)
        kw_m["label"] = "Mean value"
        # Extract options from kwargs
        for k in util.denone(kw.get("MeanOptions", {})):
            # Override the default option.
            if kw["MeanOptions"][k] is not None:
                kw_m[k] = kw["MeanOptions"][k]
        # Option whether or not to plot mean as vertical line.
        if kw.get("PlotMean", True):
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
                if np.isnan(vt) or vt in [None, False]:
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
        # Initialize options for std plot
        kw_s = DBPlotOpts(color='b', lw=2, zorder=5)
        # Extract options from kwargs
        for k in util.denone(kw.get("StDevOptions", {})):
            # Override the default option.
            if kw["StDevOptions"][k] is not None:
                kw_s[k] = kw["StDevOptions"][k]
        # Check whether or not to plot it
        if ksig and len(I) > 2 and kw.get("PlotSigma", True):
            # Check for single number or list
            if type(ksig).__name__ in ['ndarray', 'list', 'tuple']:
                # Separate lower and upper limits
                vmin = vmu - ksig[0]*vstd
                vmax = vmu + ksig[1]*vstd
            else:
                # Use as a single number
                vmin = vmu - ksig*vstd
                vmax = vmu + ksig*vstd
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
        # Initialize options for delta plot
        kw_d = DBPlotOpts(color="r", ls="--", lw=1.0, zorder=3)
        # Extract options from kwargs
        for k in util.denone(kw.get("DeltaOptions", {})):
            # Override the default option.
            if kw["DeltaOptions"][k] is not None:
                kw_d[k] = kw["DeltaOptions"][k]
        # Check whether or not to plot it
        if dc:
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
        if DBT:
            # Error in coeff
            lx = u'\u0394%s' % coeff
        else:
            # Just the value
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
        xlbl = lx if xlbl is None else xlbl
        ylbl = ly if ylbl is None else ylbl
        # Check for flipping
        if not q_vert:
            xlbl, ylbl = ylbl, xlbl
        # Labels.
        h['x'] = plt.xlabel(xlbl)
        h['y'] = plt.ylabel(ylbl)
        # Correct the font
        _set_font(h['x'])
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
            # Check for deltas
            if DBT:
                # Form: mu(DCA) = 0.0204
                klbl = (u'\u03bc(\u0394%s)' % coeff)
            else:
                # Form: CA = 0.0204
                klbl = (u'%s' % coeff)
            # Check for option
            olbl = kw.get("MuLabel", klbl)
            # Use non-default user-specified value
            if olbl is not None:
                klbl = olbl
            # Insert value
            lbl = ('%s = %s' % (klbl, flbl)) % vmu
            # Create the handle.
            h['mu'] = plt.text(
                0.99, yu, lbl,
                color=kw_m['color'],
                horizontalalignment='right', verticalalignment='top',
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
                horizontalalignment='left', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font
            _set_font(h['d'])
        # Make a label for the standard deviation
        qsig = (
            (ksig and kw.get("ShowSigma", True)) or
            kw.get("ShowSigma", False))
        if len(I) > 2 and qsig:
            # Printf-style flag
            flbl = kw.get("SigmaFormat", "%.4f")
            # Check for deltas
            if DBT:
                # Form: sigma(DCA) = 0.0204
                klbl = (u'\u03c3(\u0394%s)' % coeff)
            else:
                # Form: sigma(CA) = 0.0204
                klbl = (u'\u03c3(%s)' % coeff)
            # Check for option
            olbl = kw.get("SigmaLabel", klbl)
            # Use non-default user-specified value
            if olbl is not None:
                klbl = olbl
            # Insert value
            lbl = ('%s = %s' % (klbl, flbl)) % vstd
            # Create the handle.
            h['sig'] = plt.text(
                0.01, yu, lbl, color=kw_s.get_opt('color', 1),
                horizontalalignment='left', verticalalignment='top',
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
                horizontalalignment='right', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font
            _set_font(h['t'])
        # Output.
        return h

    # Plot a sweep of one or more coefficients
    def PlotHist(self, coeff, I, **kw):
        r"""Plot a histogram over several cases

        :Call:
            >>> h = DBi.PlotValueHist(coeff, I, **kw)
        :Inputs:
            *DBi*: :class:`cape.cfdx.databook.DataBookComp`
                An individual item data book
            *coeff*: :class:`str`
                Coefficient being plotted
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indexes of cases to include in sweep
        :Keyword Arguments:
            * See :func:`cape.cfdx.databook.DataBookComp.PlotHistBase`
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2016-04-04 ``@ddalle``: v1.0
        """
        # Call base function with no modifications to defaults
        return self.PlotHistBase(coeff, I, **kw)

    # Plot a sweep of one or more coefficients
    def PlotRangeHistBase(self, coeff, I, **kw):
        r"""Plot a range histogram of one coefficient over several cases

        :Call:
            >>> h = DBi.PlotRangeHistBase(coeff, I, **kw)
        :Inputs:
            *DBi*: :class:`cape.cfdx.databook.DataBookComp`
                An individual item data book
            *coeff*: :class:`str`
                Coefficient being plotted
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indexes of cases to include in sweep
        :Keyword Arguments:
            *FigureWidth*: :class:`float`
                Figure width
            *FigureHeight*: :class:`float`
                Figure height
            *Label*: {*comp*} | :class:`str`
                Manually specified label
            *Target*: :class:`DataBookComp` | :class:`list`
                Target database or list thereof
            *TargetValue*: :class:`float` | :class:`list`\ [:class:`float`]
                Target or list of target values
            *TargetLabel*: :class:`str` | :class:`list` (:class:`str`)
                Legend label(s) for target(s)
            *StDev*:  {``3.6863``} | ``None`` | :class:`float`
                Multiple of iterative history standard deviation to plot
            *HistOptions*: :class:`dict`
                Plot options for the primary histogram
            *StDevOptions*: :class:`dict`
                Dictionary of plot options for the standard deviation plot
            *DeltaOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for reference range plot
            *TargetOptions*: :class:`dict`
                Options passed to :func:`plt.plot` for target value lines
            *OutlierSigma*: {``3.6863``} | :class:`float`
                Standard deviation multiplier for determining outliers
            *ShowMu*: :class:`bool`
                Option to print value of mean
            *ShowSigma*: :class:`bool`
                Option to print value of standard deviation
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
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: v1.0
            * 2015-12-14 ``@ddalle``: Added error bars
            * 2016-04-04 ``@ddalle``: Moved from point sensor to data book
        """
       # -----------
       # Preparation
       # -----------
        # Make sure the plotting modules are present.
        plt = import_pyplot()
        # Figure dimensions
        fw = kw.get('FigureWidth', 6)
        fh = kw.get('FigureHeight', 4.5)
       # -----------------
       # Statistics/Values
       # -----------------
        # Check for a target
        DBT = kw.get("target", kw.get("DBT", kw.get("Target")))
        # Figure out the "component" name
        compo = self.comp
        # Check for a point or group
        pt = getattr(self, "name", getattr(self, "pt", None))
        # Form total component name
        if pt:
            # Add component and point name
            comp = "%s.%s" % (compo, pt)
        else:
            # Just a component (like STACK_No_Base)
            comp = compo
        # Extract the values or statistics, as appropriate
        if DBT is None:
            # Extract the values
            V = self[coeff][I]
        elif type(DBT).__name__ in ["list", "ndarray"]:
            # Loop through lists until found
            for DBTc in DBT:
                # Attempt to calculate statistics
                S = self.GetDeltaStats(DBTc, comp, coeff, I)
                # Extract just the deltas
                V = S["delta"]
                V = V[np.logical_not(np.isnan(V))]
                # Exit as soon as we find at least one matching case
                if len(V) > 0:
                    break
                # (If no targets match, just use the last)
        else:
            # Get statistics
            S = self.GetDeltaStats(DBT, comp, coeff, I)
            # Extract just the deltas
            V = S["delta"]
            V = V[np.logical_not(np.isnan(V))]
        # Get ranges (absolute values)
        R = np.abs(V)
        # Calculate basic statistics
        vmu = np.mean(R)
        vstd = vmu/1.128
        # Check for outliers ...
        ostd = kw.get('OutlierSigma', 3.6863)
        # Apply outlier tolerance
        if ostd:
            # Find indices of cases that are within outlier range
            J = np.abs(R)/vmu <= ostd
            # Recompute statistics
            vmu = np.mean(R[J])
            vstd = vmu/1.128
       # ------------
       # More Options
       # ------------
        # Uncertainty options
        ksig = kw.get('StDev', 3.6863)
        # Reference delta
        dc = kw.get('Delta', 0.0)
        # Target values and labels
        vtarg = kw.get('TargetValue')
        ltarg = kw.get('TargetLabel')
        # Convert target values to list
        if vtarg in [None, False]:
            vtarg = []
        elif type(vtarg).__name__ not in ['list', 'tuple', 'ndarray']:
            vtarg = [vtarg]
        # Create appropriate target list for
        if type(ltarg).__name__ not in ['list', 'tuple', 'ndarray']:
            ltarg = [ltarg]
       # --------
       # Plotting
       # --------
        # Initialize dictionary of handles.
        h = {}
       # --------------
       # Histogram Plot
       # --------------
        # Initialize plot options for histogram.
        kw_h = DBPlotOpts(
            facecolor='c',
            zorder=2,
            align='left',
            bins=20)
        # Apply *Label* option if present
        lbl = kw.get("Label")
        if lbl:
            kw_h["label"] = lbl
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
            if type(r).__name__ in ['ndarray', 'list', 'tuple']:
                # Separate lower and upper limits
                vmin = vmu + r[0]*vstd
                vmax = vmu + r[1]*vstd
            else:
                # Use as a single number
                vmin = 0
                vmax = r*vstd
            # Overwrite any range option in *kw_h*
            kw_h['range'] = (vmin, vmax)
        # Plot the histogram.
        h['hist'] = plt.hist(R, **kw_h)
       # ------------
       # Axes Handles
       # ------------
        # Get the figure and axes.
        h['fig'] = plt.gcf()
        h['ax'] = plt.gca()
        ax = h['ax']
        # Determine whether or not the distribution is normed
        q_normed = kw_h.get("normed", kw_h.get("density", False))
        # Determine whether or not the bars are vertical
        q_vert = kw_h.get("orientation", "vertical") == "vertical"
        # Get current axis limits
        if q_vert:
            # Current limits
            xmin, xmax = ax.get_xlim()
            pmin, pmax = ax.get_ylim()
            # Avoid negative ranges
            xmin = 0.0
            ax.set_xlim(xmin, xmax)
        else:
            xmin, xmax = ax.get_ylim()
            pmin, pmax = ax.get_xlim()
            # Avoid negative ranges
            xmin = 0.0
            ax.set_ylim(xmin, xmax)
       # -------------
       # Gaussian Plot
       # -------------
        # Initialize options for guassian plot
        kw_g = DBPlotOpts(color='navy', lw=1.5, zorder=7)
        kw_g["label"] = "Normal distribution"
        # Extract options from kwargs
        for k in util.denone(kw.get("GaussianOptions", {})):
            # Override the default option.
            if kw["GaussianOptions"][k] is not None:
                kw_g[k] = kw["GaussianOptions"][k]
        # Check whether or not to plot it
        if q_normed and kw.get("PlotGaussian"):
            # Lookup probabilities
            xval = np.linspace(0, xmax, 101)
            # Compute Gaussian distribution
            yval = 1/(vstd*np.sqrt(np.pi))*np.exp(-0.25*(xval/vstd)**2)
            # Check orientation
            if q_vert:
                # Plot a vertical line for the mean.
                h['mean'] = plt.plot(xval, yval, **kw_g)
            else:
                # Plot a horizontal line for th emean.
                h['mean'] = plt.plot(yval, xval, **kw_g)
       # ---------
       # Mean Plot
       # ---------
        # Initialize options for mean plot
        kw_m = DBPlotOpts(color='k', lw=2, zorder=6)
        kw_m["label"] = "Mean value"
        # Extract options from kwargs
        for k in util.denone(kw.get("MeanOptions", {})):
            # Override the default option.
            if kw["MeanOptions"][k] is not None:
                kw_m[k] = kw["MeanOptions"][k]
        # Option whether or not to plot mean as vertical line.
        if kw.get("PlotMean", False):
            # Check orientation
            if q_vert:
                # Plot a vertical line for the mean.
                h['mean'] = plt.plot([vmu, vmu], [pmin, pmax], **kw_m)
            else:
                # Plot a horizontal line for the mean.
                h['mean'] = plt.plot([pmin, pmax], [vmu, vmu], **kw_m)
       # -----------
       # Target Plot
       # -----------
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
        # Option whether or not to plot targets
        if vtarg is not None and len(vtarg) > 0:
            # Loop through target values
            for i in range(len(vtarg)):
                # Select the value
                vt = vtarg[i]
                # Check for NaN or None
                if np.isnan(vt) or vt in [None, False]:
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
        # Initialize options for std plot
        kw_s = DBPlotOpts(color='b', lw=2, zorder=5)
        # Extract options from kwargs
        for k in util.denone(kw.get("StDevOptions", {})):
            # Override the default option.
            if kw["StDevOptions"][k] is not None:
                kw_s[k] = kw["StDevOptions"][k]
        # Check whether or not to plot it
        if ksig and len(I) > 2 and kw.get("PlotSigma", True):
            # Set value
            vs = ksig*vstd
            # Check orientation
            if q_vert:
                # Plot a vertical line for the min and max
                h['std'] = plt.plot([vs, vs], [pmin, pmax], **kw_s)
            else:
                # Plot a horizontal line for the min and max
                h['std'] = plt.plot([pmin, pmax], [vs, vs], **kw_s)
       # ----------
       # Delta Plot
       # ----------
        # Initialize options for delta plot
        kw_d = DBPlotOpts(color="r", ls="--", lw=1.0, zorder=3)
        # Extract options from kwargs
        for k in util.denone(kw.get("DeltaOptions", {})):
            # Override the default option.
            if kw["DeltaOptions"][k] is not None:
                kw_d[k] = kw["DeltaOptions"][k]
        # Check whether or not to plot it
        if dc:
            # Check orientation
            if q_vert:
                # Plot vertical lines for the reference length
                h['delta'] = plt.plot([dc, dc], [pmin, pmax], **kw_d)
            else:
                # Plot horizontal lines for reference length
                h['delta'] = plt.plot([pmin, pmax], [dc, dc], **kw_d)
       # ----------
       # Formatting
       # ----------
        # Default value-axis label
        if DBT:
            # Error in coeff
            lx = u'\u0394%s' % coeff
        else:
            # Just the value
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
        # Correct the font
        _set_font(h['x'])
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
            # Check for deltas
            if DBT:
                # Form: mu(DCA) = 0.0204
                klbl = (u'\u03bc(\u0394%s)' % coeff)
            else:
                # Form: CA = 0.0204
                klbl = (u'%s' % coeff)
            # Check for option
            olbl = kw.get("MuLabel", klbl)
            # Use non-default user-specified value
            if olbl is not None:
                klbl = olbl
            # Insert value
            lbl = ('%s = %s' % (klbl, flbl)) % vmu
            # Create the handle.
            h['mu'] = plt.text(
                0.99, yu, lbl,
                color=kw_m['color'],
                horizontalalignment='right', verticalalignment='top',
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
                horizontalalignment='left', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font
            _set_font(h['d'])
        # Make a label for the standard deviation.
        if len(I) > 2 and (
                (ksig and kw.get("ShowSigma", True)) or
                kw.get("ShowSigma", False)):
            # Printf-style flag
            flbl = kw.get("SigmaFormat", "%.4f")
            # Check for deltas
            if DBT:
                # Form: sigma(DCA) = 0.0204
                klbl = (u'\u03c3(\u0394%s)' % coeff)
            else:
                # Form: sigma(CA) = 0.0204
                klbl = (u'\u03c3(%s)' % coeff)
            # Check for option
            olbl = kw.get("SigmaLabel", klbl)
            # Use non-default user-specified value
            if olbl is not None:
                klbl = olbl
            # Insert value
            lbl = ('%s = %s' % (klbl, flbl)) % vstd
            # Create the handle.
            h['sig'] = plt.text(
                0.01, yu, lbl,
                color=kw_s.get_opt('color', 1),
                horizontalalignment='left', verticalalignment='top',
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
                horizontalalignment='right', verticalalignment='top',
                transform=h['ax'].transAxes)
            # Correct the font
            _set_font(h['t'])
        # Output.
        return h

    # Plot a sweep of one or more coefficients
    def PlotRangeHist(self, coeff, I, **kw):
        r"""Plot a range histogram over several cases

        :Call:
            >>> h = DBi.PlotRangeHist(coeff, I, **kw)
        :Inputs:
            *DBi*: :class:`cape.cfdx.databook.DataBookComp`
                An individual item data book
            *coeff*: :class:`str`
                Coefficient being plotted
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indexes of cases to include in sweep
        :Keyword Arguments:
            * See :func:`cape.cfdx.databook.DataBookComp.PlotHistBase`
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2016-04-04 ``@ddalle``: v1.0
        """
        # Call base function with no modifications to defaults
        return self.PlotRangeHistBase(coeff, I, **kw)
  # >


# Data book for an individual component
class FMDataBook(DataBookComp):
    r"""Individual force & moment component data book

    This class is derived from :class:`cape.cfdx.databook.DataBookComp`.

    :Call:
        >>> DBi = FMDataBookComp(comp, cntl, targ=None, **kw)
    :Inputs:
        *comp*: :class:`str`
            Name of the component
        *cntl*: :class:`Cntl`
            CAPE control class instance
        *targ*: {``None``} | :class:`str`
            If used, read a duplicate data book as a target named *targ*
        *check*: ``True`` | {``False``}
            Whether or not to check LOCK status
        *lock*: ``True`` | {``False``}
            If ``True``, wait if the LOCK file exists
    :Outputs:
        *DBi*: :class:`cape.cfdx.databook.FMDataBook`
            An individual component data book
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
        * 2014-12-22 ``@ddalle``: v1.0
        * 2016-06-27 ``@ddalle``: Added target option for using other folders
    """
  # ========
  # Config
  # ========
  # <
    # Initialization method
    def __init__(self, comp, cntl, targ=None, check=False, lock=False, **kw):
        """Initialization method

        :Versions:
            * 2014-12-21 ``@ddalle``: v1.0
            * 2022-04-13 ``@ddalle``: v2.0; use *cntl*
        """
        # Unpack *cntl*
        x = cntl.x
        opts = cntl.opts
        # Save relevant inputs
        self.x = x
        self.opts = opts
        self.cntl = cntl
        self.comp = comp
        self.name = comp
        self.sources = {}
        # Root directory
        self.RootDir = kw.get("RootDir", os.getcwd())

        # Get the directory.
        if targ is None:
            # Primary data book directory
            fdir = opts.get_DataBookFolder()
        else:
            # Secondary data book directory
            fdir = opts.get_TargetDataBookDir(targ)

        # Construct the file name.
        fcomp = 'aero_%s.csv' % comp
        # Folder name for compatibility.
        fdir = fdir.replace("/", os.sep)
        fdir = fdir.replace("\\", os.sep)
        # Construct the full file name.
        fname = os.path.join(fdir, fcomp)
        # Save the file name.
        self.fname = fname
        self.fdir = fdir

        # Process columns
        self.ProcessColumns()

        # Read the file or initialize empty arrays.
        self.Read(self.fname, check=check, lock=lock)

        # Save the target translations
        self.targs = opts.get_CompTargets(comp)
        # Divide columns into parts
        self.DataCols = opts.get_DataBookDataCols(comp)

    # Command-line representation
    def __repr__(self):
        r"""Representation method

        :Versions:
            * 2014-12-27 ``@ddalle``: v1.0
        """
        # Initialize string
        lbl = "<FMDataBook %s, " % self.comp
        # Add the number of conditions.
        lbl += "nCase=%i>" % self.n
        # Output
        return lbl
    # String conversion
    __str__ = __repr__
  # >

  # ======
  # Data
  # ======
  # <
    # Get a value
    def GetCoeff(self, comp, coeff, I, **kw):
        r"""Get a coefficient value for one or more cases

        :Call:
            >>> v = DBT.GetCoeff(comp, coeff, i)
            >>> V = DBT.GetCoeff(comp, coeff, I)
        :Inputs:
            *DBT*: :class:`cape.cfdx.databook.TargetDataBook`
                Instance of the Cape data book target class
            *comp*: :class:`str`
                Component whose coefficient is being plotted
            *coeff*: :class:`str`
                Coefficient being plotted
            *i*: :class:`int`
                Individual case/entry index
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indexes of cases to include in sweep
        :Outputs:
            *v*: :class:`float`
                Scalar value from the appropriate column
            *V*: :class:`np..ndarray`
                Array of values from the appropriate column
        :Versions:
            * 2018-02-12 ``@ddalle``: v1.0
        """
        # Check for patch delimiter
        if "/" in comp:
            # Format: Cp_ports.P001
            compo, pt = comp.split("/")
        elif "." in comp:
            # Format: Cp_ports/P001
            compo, pt = comp.split(".")
        else:
            # Only comp given; use total of point names
            compo = comp
            pt = None
        # Check the component
        try:
            # Check if the component equals *compo*
            if self.comp != compo:
                raise ValueError(
                    ("DataBook component is '%s'; " % self.comp) +
                    ("cannot match '%s'" % compo))
        except AttributeError:
            # Could not find component
            pass
        # Check the point/name
        try:
            # Try the *name* attribute
            name = self.name
        except AttributeError:
            # Try the *pt*
            try:
                name = self.pt
            except AttributeError:
                name = None
        # Get point if applicable
        if pt is not None:
            # Compare the point
            if pt != name:
                raise ValueError(
                    ("DataBook name is '%s'; " % name) +
                    ("cannot match '%s'" % pt))
        # Get the value
        return self[coeff][I]

    # Transform force or moment reference frame
    def TransformFM(self, topts, mask=None):
        r"""Transform force and moment coefficients

        Available transformations and their parameters are

            * "Euler123": "phi", "theta", "psi"
            * "Euler321": "psi", "theta", "phi"
            * "ScaleCoeffs": "CA", "CY", "CN", "CLL", "CLM", "CLN"

        Other variables (columns) in the databook are used to specify
        values to use for the transformation variables.  For example,

            .. code-block:: python

                topts = {
                    "Type": "Euler321",
                    "psi": "Psi",
                    "theta": "Theta",
                    "phi": "Phi",
                }

        will cause this function to perform a reverse Euler 3-2-1
        transformation using *dbc["Psi"]*, *dbc["Theta"]*, and
        *dbc["Phi"]* as the angles.

        Coefficient scaling can be used to fix incorrect reference areas
        or flip axes. The default is actually to flip *CLL* and *CLN*
        due to the transformation from CFD axes to standard flight
        dynamics axes.

            .. code-block:: python

                topts = {
                    "Type": "ScaleCoeffs",
                    "CLL": -1.0,
                    "CLN": -1.0,
                }

        :Call:
            >>> dbc.TransformFM(topts, mask=None)
        :Inputs:
            *dbc*: :class:`DataBookComp`
                Instance of the force and moment class
            *topts*: :class:`dict`
                Dictionary of options for the transformation
            *mask*: {``None``} | :class:`np.ndarray`\ [:class:`int`]
                Optional subset of cases to transform
        :Versions:
            * 2021-11-18 ``@ddalle``: v1.0
        """
        # Get the transformation type.
        ttype = topts.get("Type", "")
        # Default mask
        if mask is None:
            mask = np.arange(self["CA"].size)
        # Check it.
        if ttype in ["Euler321", "Euler123"]:
            # Get the angle variable names.
            # Use same as default in case it's obvious what they should be.
            kph = topts.get('phi', 0.0)
            kth = topts.get('theta', 0.0)
            kps = topts.get('psi', 0.0)
            # Extract roll
            if isinstance(kph, (float, np.float)):
                # Singleton
                phi = kph*deg * np.ones_like(mask)
            elif isinstance(kph, np.ndarray):
                # Directly specified value(s)
                phi = kph*deg
            elif kph.startswith('-'):
                # Negative roll angle
                phi = -self[kph[1:]][mask]*deg
            else:
                # Positive roll
                phi = self[kph][mask]*deg
            # Extract pitch
            if isinstance(kth, (float, np.float)):
                # Singleton
                theta = kth*deg * np.ones_like(mask)
            elif isinstance(kth, np.ndarray):
                # Directly specified value(s)
                theta = kth*deg
            elif kth.startswith('-'):
                # Negative pitch
                theta = -self[kth[1:]][mask]*deg
            else:
                # Positive pitch
                theta = self[kth][mask]*deg
            # Extract yaw
            if isinstance(kps, (float, np.float)):
                # Singleton
                psi = kps*deg * np.ones_like(mask)
            elif isinstance(kps, np.ndarray):
                # Directly specified value(s)
                psi = kps*deg
            elif kps.startswith('-'):
                # Negative yaw
                psi = -self[kps[1:]][mask]*deg
            else:
                # Positive yaw
                psi = self[kps][mask]*deg
            # Loop through cases
            for j, (phj, thj, psj) in enumerate(zip(phi, theta, psi)):
                # Sines and cosines
                cph = np.cos(phj)
                cth = np.cos(thj)
                cps = np.cos(psj)
                sph = np.sin(phj)
                sth = np.sin(thj)
                sps = np.sin(psj)
                # Make the matrices
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
                # Area transformations
                if "Ay" in self:
                    # Assemble area vector
                    Ac = np.array(
                        [self["Ax"][j], self["Ay"][j], self["Az"][j]])
                    # Transform
                    Ab = np.dot(R, Ac)
                    # Reset
                    self["Ax"][j] = Ab[0]
                    self["Ay"][j] = Ab[1]
                    self["Az"][j] = Ab[2]
                # Force transformations
                # Loop through suffixes
                for s in ["", "p", "vac", "v", "m"]:
                    # Construct force coefficient names
                    cx = "CA" + s
                    cy = "CY" + s
                    cz = "CN" + s
                    # Check if the coefficient is present
                    if cy in self:
                        # Assemble forces
                        Fc = np.array([self[cx][j], self[cy][j], self[cz][j]])
                        # Transform
                        Fb = np.dot(R, Fc)
                        # Reset
                        self[cx][j] = Fb[0]
                        self[cy][j] = Fb[1]
                        self[cz][j] = Fb[2]
                    # Construct moment coefficient names
                    cx = "CLL" + s
                    cy = "CLM" + s
                    cz = "CLN" + s
                    # Check if the coefficient is present
                    if cy in self:
                        # Assemble moment vector
                        Mc = np.array([self[cx][j], self[cy][j], self[cz][j]])
                        # Transform
                        Mb = np.dot(R, Mc)
                        # Reset
                        self[cx][j] = Mb[0]
                        self[cy][j] = Mb[1]
                        self[cz][j] = Mb[2]
        elif ttype in ["ScaleCoeffs"]:
            # Loop through coefficients.
            for c in topts:
                # Get the value.
                k = topts[c]
                # Check if it's a number.
                if not isinstance(k, (float, int, np.float)):
                    # Assume they meant to flip it.
                    k = -1.0
                # Loop through suffixes
                for s in ["", "p", "vac", "v", "m"]:
                    # Construct overall name
                    cc = c + s
                    # Check if it's present
                    if cc in self:
                        self[cc] = k*self[cc]
        else:
            raise IOError(
                "Transformation type '%s' is not recognized." % ttype)
  # >

  # ======
  # Read
  # ======
  # <
    def ReadCase(self, comp):
        r"""Read a :class:`CaseFM` object

        :Call:
            >>> fm = DB.ReadCaseFM(comp)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.FMDataBook`
                Instance of data book class
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *fm*: :class:`cape.cfdx.databook.CaseFM`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseFM(comp)

    # Read residual history class
    def ReadCaseResid(self):
        r"""Read a :class:`CaseResid` object

        :Call:
            >>> H = DB.ReadCaseResid()
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBookComp`
                Instance of data book class
        :Outputs:
            *H*: :class:`cape.cfdx.databook.CaseResid`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseResid()
  # >

  # ======
  # Write
  # ======
  # <
    # Update or add an entry for one component
    def UpdateCaseDB(self, i, j, comp):
        r"""Update or add a case to a data book

        The history of a run directory is processed if either one of
        three criteria are met.

            1. The case is not already in the data book
            2. The most recent iteration is greater than the data book
               value
            3. The number of iterations used to create statistics has
               changed

        :Call:
            >>> n = DB.UpdateCaseComp(i, comp)
        :Inputs:
            *DB*: :class:`pyFun.databook.DataBook`
                Instance of the data book class
            *i*: :class:`int`
                RunMatrix index
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *n*: ``0`` | ``1``
                How many updates were made
        :Versions:
            * 2014-12-22 ``@ddalle``: v1.0
            * 2017-04-12 ``@ddalle``: Modified to work one component
            * 2017-04-23 ``@ddalle``: Added output
        """
        # Get the first data book component.
        DBc = self
        # Get the current iteration number
        nIter = self.cntl.GetCurrentIter(i)
        # Get the number of iterations used for statutils.
        nStats = self.opts.get_DataBookNStats(comp)
        # Get the iteration at which statistics can begin.
        nMin = self.opts.get_DataBookNMin(comp)
        # Maximum number of iterations allowed
        nMaxStats = self.opts.get_DataBookNMaxStats(comp)
        # Limit max stats if instructed to do so
        if nMaxStats is None:
            # No max
            nMax = None
        else:
            # Specified max, but don't use data before *nMin*
            nMax = min(nIter - nMin, nMaxStats)
        # Read residual
        H = self.ReadCaseResid()
       # --- Read Iterative History ---
        # Get component (note this automatically defaults to *comp*)
        compID = self.opts.get_DataBookCompID(comp)
        # Check for multiple components
        if type(compID).__name__ in ['list', 'ndarray']:
            # Read the first component
            FM = self.ReadCase(compID[0])
            # Loop through remaining components
            for compi in compID[1:]:
                # Check for minus sign
                if compi.startswith('-'):
                    # Subtract the component
                    FM -= self.ReadCase(compi.lstrip('-'))
                else:
                    # Add in the component
                    FM += self.ReadCase(compi)
        else:
            # Read the iterative history for single component
            FM = self.ReadCase(compID)
        # List of transformations
        tcomp = self.opts.get_DataBookTransformations(comp)
        tcomp = list(tcomp)
        # Special transformation to reverse *CLL* and *CLN*
        tflight = {
            "Type": "ScaleCoeffs",
            "CLL": -1.0,
            "CLN": -1.0
        }
        # Check for ScaleCoeffs
        for tj in tcomp:
            # Skip if not a "ScaleCoeffs"
            if tj.get("Type") != "ScaleCoeffs":
                continue
            # Use it if we have either *CLL* or *CLM*
            if "CLL" in tj or "CLN" in tj:
                break
        else:
            # If we didn't find a match, append *tflight*
            tcomp.append(tflight)
        # Save the Lref, current MRP to any "ShiftMRP" transformations
        for topts in tcomp:
            # Get type
            ttyp = topts.get("Type")
            # Only apply to "ShiftMRP"
            if ttyp == "ShiftMRP":
                # Use a copy to avoid changing cntl.opts
                topts = dict(topts)
                # Component to use for current MRP
                compID = self.cntl.opts.get_DataBookCompID(comp)
                if isinstance(compID, list):
                    compID = compID[0]
                # Reset points for default *FromMRP*
                self.cntl.opts.reset_Points()
                # Use MRP prior to transfformations as default *FromMRP*
                x0 = self.cntl.opts.get_RefPoint(comp)
                # Ensure points are calculated
                self.cntl.PreparePoints(i)
                # Use post-transformation MRP as default *ToMRP*
                x1 = self.cntl.opts.get_RefPoint(comp)
                # Get current Lref
                Lref = self.cntl.opts.get_RefLength(comp)
                # Set those as defaults in transformation
                x0 = topts.setdefault("FromMRP", x0)
                x1 = topts.setdefault("ToMRP", x1)
                topts.setdefault("RefLength", Lref)
                # Expand if *x0* is a string
                topts["FromMRP"] = self.cntl.opts.expand_Point(x0)
                topts["ToMRP"] = self.cntl.opts.expand_Point(x1)
            # Apply the transformation.
            FM.TransformFM(topts, self.x, i)

        # Process the statistics.
        s = FM.GetStats(nStats, nMax)
        # Get the corresponding residual drop
        if 'nOrders' in DBc:
            nOrders = H.GetNOrders(s['nStats'])

        # Save the data.
        if j is None:
            # Add to the number of cases.
            DBc.n += 1
            # Append trajectory values.
            for k in self.x.cols:
                # Append
                DBc[k] = np.append(DBc[k], self.x[k][i])
            # Append values.
            for c in DBc.DataCols:
                if c in s:
                    DBc[c] = np.append(DBc[c], s[c])
                else:
                    DBc[c] = np.append(DBc[c], np.nan)
            # Append residual drop.
            if 'nOrders' in DBc:
                DBc['nOrders'] = np.hstack((DBc['nOrders'], [nOrders]))
            # Append iteration counts.
            if 'nIter' in DBc:
                DBc['nIter']  = np.hstack((DBc['nIter'], [nIter]))
            if 'nStats' in DBc:
                DBc['nStats'] = np.hstack((DBc['nStats'], [s['nStats']]))
        else:
            # Save updated trajectory values
            for k in DBc.xCols:
                # Append to that column
                DBc[k][j] = self.x[k][i]
            # Update data values.
            for c in DBc.DataCols:
                DBc[c][j] = s[c]
            # Update the other statistics.
            if 'nOrders' in DBc:
                DBc['nOrders'][j] = nOrders
            if 'nIter' in DBc:
                DBc['nIter'][j]   = nIter
            if 'nStats' in DBc:
                DBc['nStats'][j]  = s['nStats']
  # >


# Data book for an individual component
class PropDataBook(DataBookComp):
    r"""Individual generic-property component data book

    This class is derived from :class:`cape.cfdx.databook.DataBookComp`.

    :Call:
        >>> dbk = PropDataBook(comp, cntl, targ=None, **kw)
    :Inputs:
        *comp*: :class:`str`
            Name of the component
        *cntl*: :class:`Cntl`
            CAPE control instance
        *targ*: {``None``} | :class:`str`
            If used, read a duplicate data book as a target named *targ*
        *check*: ``True`` | {``False``}
            Whether or not to check LOCK status
        *lock*: ``True`` | {``False``}
            If ``True``, wait if the LOCK file exists
    :Outputs:
        *dbk*: :class:`PropDataBook`
            An individual generic-property component data book
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
        * 2014-12-22 ``@ddalle``: v1.0 (:class:`DBComp`)
        * 2016-06-27 ``@ddalle``: v1.1
        * 2022-04-08 ``@ddalle``: v1.0
    """
  # ========
  # Config
  # ========
  # <
    # Initialization method
    def __init__(self, comp, cntl, targ=None, check=False, **kw):
        """Initialization method

        :Versions:
            * 2014-12-21 ``@ddalle``: v1.0
        """
        # Unpack *cntl*
        x = cntl.x
        opts = cntl.opts
        # Save relevant inputs
        self.x = x
        self.opts = opts
        self.cntl = cntl
        self.comp = comp
        self.name = comp
        self.sources = {}
        # Root directory
        self.RootDir = kw.get("RootDir", os.getcwd())
        # Opitons
        lock = kw.get("lock", False)

        # Get the directory.
        if targ is None:
            # Primary data book directory
            fdir = opts.get_DataBookFolder()
        else:
            # Secondary data book directory
            fdir = opts.get_TargetDataBookDir(targ)

        # Construct the file name.
        fcomp = 'prop_%s.csv' % comp
        # Folder name for compatibility.
        fdir = fdir.replace("/", os.sep)
        fdir = fdir.replace("\\", os.sep)
        # Construct the full file name.
        fname = os.path.join(fdir, fcomp)
        # Save the file name.
        self.fname = fname
        self.fdir = fdir

        # Process columns
        self.ProcessColumns()

        # Read the file or initialize empty arrays.
        self.Read(self.fname, check=check, lock=lock)

        # Save the target translations
        self.targs = opts.get_CompTargets(comp)
        # Divide columns into parts
        self.DataCols = opts.get_DataBookDataCols(comp)

    # Command-line representation
    def __repr__(self):
        r"""Representation method

        :Versions:
            * 2014-12-27 ``@ddalle``: v1.0
        """
        # Initialize string
        lbl = "<PropDataBook %s, " % self.comp
        # Add the number of conditions.
        lbl += "nCase=%i>" % self.n
        # Output
        return lbl
    # String conversion
    __str__ = __repr__
  # >

  # ======
  # Read
  # ======
  # <
    # Update or add an entry for one component
    def ReadCase(self, comp):
        r"""Read a :class:`CaseProp` object

        :Call:
            >>> prop = DB.ReadCaseProp(comp)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.PropDataBook`
                Instance of data book class
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *prop*: :class:`cape.cfdx.databook.CaseProp`
                Generic-property iterative history instance
        :Versions:
            * 2022-04-08 ``@ddalle``: v1.0
        """
        # Read CaseResid object from PWD
        return CaseProp(comp)
    # >

  # ======
  # Write
  # ======
  # <
    # Update or add an entry for one component
    def UpdateCaseDB(self, i, j, comp):
        r"""Update or add a case to a data book

        The history of a run directory is processed if either one of
        three criteria are met.

            1. The case is not already in the data book
            2. The most recent iteration is greater than the data book
               value
            3. The number of iterations used to create statistics has
               changed

        :Call:
            >>> n = DB.UpdateCaseComp(i, comp)
        :Inputs:
            *DB*: :class:`pyFun.databook.DataBook`
                Instance of the data book class
            *i*: :class:`int`
                RunMatrix index
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *n*: ``0`` | ``1``
                How many updates were made
        :Versions:
            * 2014-12-22 ``@ddalle``: v1.0
            * 2017-04-12 ``@ddalle``: Modified to work one component
            * 2017-04-23 ``@ddalle``: Added output
        """
        # Get the first data book component.
        DBc = self
        # Get the current iteration number
        nIter = self.cntl.GetCurrentIter(i)
        # Get the number of iterations used for statutils.
        nStats = self.opts.get_DataBookNStats(comp)
        # Get the iteration at which statistics can begin.
        nMin = self.opts.get_DataBookNMin(comp)
        # Maximum number of iterations allowed
        nMaxStats = self.opts.get_DataBookNMaxStats(comp)
        # Limit max stats if instructed to do so
        if nMaxStats is None:
            # No max
            nMax = None
        else:
            # Specified max, but don't use data before *nMin*
            nMax = min(nIter - nMin, nMaxStats)
       # --- Read Iterative History ---
        # Get component (note this automatically defaults to *comp*)
        compID = self.opts.get_DataBookCompID(comp)
        # Read the iterative history for single component
        prop = self.ReadCase(compID)
        # Process the statistics.
        s = prop.GetStats(nStats, nMax)
        # Get the corresponding residual drop
        # Save the data.
        if j is None:
            # Add to the number of cases
            DBc.n += 1
            # Append trajectory values
            for k in self.x.cols:
                # Append
                DBc[k] = np.append(DBc[k], self.x[k][i])
            # Append values
            for c in DBc.DataCols:
                if c in s:
                    DBc[c] = np.append(DBc[c], s[c])
            # Append iteration counts
            if 'nIter' in DBc:
                DBc['nIter']  = np.hstack((DBc['nIter'], [nIter]))
            if 'nStats' in DBc:
                DBc['nStats'] = np.hstack((DBc['nStats'], [s['nStats']]))
        else:
            # Save updated trajectory values
            for k in DBc.xCols:
                # Append to that column
                DBc[k][j] = self.x[k][i]
            # Update data values.
            for c in DBc.DataCols:
                DBc[c][j] = s[c]
            # Update the other statistics.
            if 'nIter' in DBc:
                DBc['nIter'][j] = nIter
            if 'nStats' in DBc:
                DBc['nStats'][j] = s['nStats']
  # >


# Data book for an individual component
class PyFuncDataBook(DataBookComp):
    r"""Individual scalar Python output component data book

    This class is derived from :class:`DataBookComp`.

    :Call:
        >>> dbk = PyFuncDataBook(comp, x, opts, funcname, **kw)
    :Inputs:
        *comp*: :class:`str`
            Name of the component
        *x*: :class:`cape.runmatrix.RunMatrix`
            RunMatrix for processing variable types
        *opts*: :class:`cape.cfdx.options.Options`
            Global pyCart options instance
        *funcname*: :class:`str`
            Name of function to execute
        *targ*: {``None``} | :class:`str`
            If used, read a duplicate data book as a target named *targ*
        *check*: ``True`` | {``False``}
            Whether or not to check LOCK status
        *lock*: ``True`` | {``False``}
            If ``True``, wait if the LOCK file exists
    :Outputs:
        *dbk*: :class:`PropDataBook`
            An individual generic-property component data book
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
        * 2014-12-22 ``@ddalle``: v1.0 (:class:`DBComp`)
        * 2016-06-27 ``@ddalle``: v1.1
        * 2022-04-10 ``@ddalle``: v1.0
    """
  # ========
  # Config
  # ========
  # <
    # Initialization method
    def __init__(self, comp, cntl, targ=None, check=False, **kw):
        """Initialization method

        :Versions:
            * 2014-12-21 ``@ddalle``: v1.0
        """
        # Save relevant inputs
        self.x = cntl.x
        self.opts = cntl.opts
        self.cntl = cntl
        self.comp = comp
        self.name = comp
        self.sources = {}
        # Root directory
        self.RootDir = kw.get("RootDir", cntl.RootDir)
        # Opitons
        lock = kw.get("lock", False)

        # Get the directory.
        if targ is None:
            # Primary data book directory
            fdir = cntl.opts.get_DataBookFolder()
        else:
            # Secondary data book directory
            fdir = cntl.opts.get_TargetDataBookDir(targ)

        # Construct the file name.
        fcomp = 'pyfunc_%s.csv' % comp
        # Folder name for compatibility.
        fdir = fdir.replace("/", os.sep)
        fdir = fdir.replace("\\", os.sep)
        # Construct the full file name.
        fname = os.path.join(fdir, fcomp)
        # Save the file name.
        self.fname = fname
        self.fdir = fdir

        # Process columns
        self.ProcessColumns()

        # Read the file or initialize empty arrays.
        self.Read(self.fname, check=check, lock=lock)

        # Get function name
        self.funcname = cntl.opts.get_DataBookFunction(comp)
        # Save the target translations
        self.targs = cntl.opts.get_CompTargets(comp)
        # Divide columns into parts
        self.DataCols = cntl.opts.get_DataBookDataCols(comp)
  # >

  # =========
  # Function
  # =========
  # <
    # Execute the function
    def ExecPyFuncDataBook(self, i):
        r"""Execute main PyFunc function and return results

        :Call:
            >>> v = db.ExecPyFuncDataBook(i)
        :Inputs:
            *db*: :class:`PyFuncDataBook`
                Databook component of type ``"PyFunc"``
            *i*: :class:`int`
                Run matrix case index
        :Outputs:
            *v*: :class:`tuple`
                Outputs from *db.funcname* in folder of case *i*
        :Versions:
            * 2022-04-13 ``@ddalle``: v1.0
        """
        # Get folder for case *i*
        frun = self.x.GetFullFolderNames(i)
        # Remember current locaiton
        fpwd = os.getcwd()
        # Use a try/catch block like @run_rootdir
        try:
            # Got to run folder
            os.chdir(self.cntl.RootDir)
            os.chdir(frun)
            # Execute the funciton
            v = self.cntl.exec_modfunction(self.funcname, (self.cntl, i))
        except Exception as e:
            # Tell user about it, but don't fail
            print(
                "    Function '%s' for case %i raised an exception:" %
                (self.funcname, i))
            traceback.print_exception(e.__class__, e, e.__traceback__, limit=2)
            # Null output
            v = None
        finally:
            # Return to original location
            os.chdir(fpwd)
        # Ensure tuple (for unpacking later)
        if v is not None and not isinstance(v, (tuple, dict)):
            v = v,
        # Output
        return v
  # >
  # ======
  # Write
  # ======
  # <

    # Update or add an entry for one component
    def UpdateCaseDB(self, i, j, comp):
        r"""Update or add a case to a data book

        The history of a run directory is processed if either one of
        three criteria are met.

            1. The case is not already in the data book
            2. The most recent iteration is greater than the data book
               value
            3. The number of iterations used to create statistics has
               changed

        :Call:
            >>> n = DB.UpdateCaseComp(i, comp)
        :Inputs:
            *DB*: :class:`pyFun.databook.DataBook`
                Instance of the data book class
            *i*: :class:`int`
                RunMatrix index
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *n*: ``0`` | ``1``
                How many updates were made
        :Versions:
            * 2014-12-22 ``@ddalle``: v1.0
            * 2017-04-12 ``@ddalle``: Modified to work one component
            * 2017-04-23 ``@ddalle``: Added output
        """
        # Get the first data book component.
        DBc = self
        # Get the current iteration number
        nIter = self.cntl.GetCurrentIter(i)
        # Get the number of iterations used for statutils.
        nStats = self.opts.get_DataBookNStats(comp)
        # Execute the appropriate function
        v = DBc.ExecPyFuncDataBook(i)
        # Check for success
        if v is None:
            return 0
        # Save the data.
        if j is None:
            # Add to the number of cases
            DBc.n += 1
            # Append trajectory values
            for k in self.x.cols:
                # Append
                DBc[k] = np.append(DBc[k], self.x[k][i])
            # Append values
            for j1, c in enumerate(DBc.DataCols):
                # Check output type from function
                if isinstance(v, dict):
                    # Get columns by name
                    vj = v[c]
                else:
                    # Get values by index
                    vj = v[j1]
                # Append to existing array
                DBc[c] = np.append(DBc[c], vj)
            # Append iteration counts
            if 'nIter' in DBc:
                DBc['nIter']  = np.hstack((DBc['nIter'], [nIter]))
            # Append iteration counts
            if 'nStats' in DBc:
                DBc['nStats']  = np.hstack((DBc['nStats'], [nStats]))
        else:
            # Save updated trajectory values
            for k in DBc.xCols:
                # Append to that column
                DBc[k][j] = self.x[k][i]
            # Update data values.
            for j1, c in enumerate(DBc.DataCols):
                # Check output type from function
                if isinstance(v, dict):
                    # Get columns by name
                    DBc[c][j] = v[c]
                else:
                    # Get values by index
                    DBc[c][j] = v[j1]
            # Update the other statistics.
            if 'nIter' in DBc:
                DBc['nIter'][j] = nIter
            if 'nStats' in DBc:
                DBc['nStats']  = nStats
  # >


# Data book for a TriqFM component
class TriqFMFaceDataBook(FMDataBook):
    r"""Force and moment component extracted from surface triangulation

    :Call:
        >>> DBF = TriqFMDataBook(x, opts, comp, RootDir=None)
    :Inputs:
        *x*: :class:`cape.runmatrix.RunMatrix`
            RunMatrix/run matrix interface
        *opts*: :class:`cape.cfdx.options.Options`
            Options interface
        *comp*: :class:`str`
            Name of TriqFM component
        *RootDir*: {``None``} | :class:`st`
            Root directory for the configuration
        *check*: ``True`` | {``False``}
            Whether or not to check LOCK status
        *lock*: ``True`` | {``False``}
            If ``True``, wait if the LOCK file exists
    :Outputs:
        *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
            Instance of TriqFM data book
    :Versions:
        * 2017-03-28 ``@ddalle``: v1.0
    """
  # ======
  # Config
  # ======
  # <
    # Initialization method
    def __init__(self, x, opts, comp, patch=None, **kw):
        """Initialization method

        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
        """
        # Save relevant inputs
        self.x = x
        self.opts = opts
        self.comp = comp
        self.patch = patch
        self.sources = {}

        # LOCK options
        check = kw.get("check", False)
        lock  = kw.get("lock",  False)

        # Default prefix
        fpre = opts.get_DataBookPrefix(comp)
        # Use name of component as default
        fpre = comp if fpre is None else fpre

        # Assemble overall component
        if patch is None:
            # Just the component
            self.name = fpre
        else:
            # Take the patch name, but ensure one occurrence of comp as prefix
            if patch.startswith(fpre):
                # Remove prefix
                name = patch[len(fpre):].lstrip('_')
            else:
                # Use the name as-is
                name = patch
            # Add the prefix (back if necessary)
            self.name = "%s_%s" % (fpre, name)

        # Save root directory
        self.RootDir = kw.get('RootDir', os.getcwd())
        # Get the data book directory
        fdir = opts.get_DataBookFolder()
        # Compatibility
        fdir = fdir.replace("/", os.sep)
        fdir = fdir.replace("\\", os.sep)
        # Save home folder
        self.fdir = fdir

        # Construct the file name
        fcomp = "triqfm_%s.csv" % self.name
        # Full file name
        fname = os.path.join(fdir, "triqfm", fcomp)
        # Save the file name
        self.fname = fname

        # Process columns
        self.ProcessColumns()

        # Read the file or initialize empty arrays
        self.Read(fname, check=check, lock=lock)

  # ======
  # Write
  # ======
  # <
    # Process a case
    def UpdateCaseDB(self, i, j, comp):
        r"""Prepare to update a TriqFM group if necessary

        :Call:
            >>> n = DBF.UpdateCase(i)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *i*: :class:`int`
                Case index
        :Outputs:
            *n*: ``0`` | ``1``
                How many updates were made
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
        """
       # -----------
       # Calculation
       # -----------
       # (
        # Get the current iteration number
        nIter = self.cntl.GetCurrentIter(i)
        # Get the number of iterations used for statutils.
        nStats = self.opts.get_DataBookNStats(comp)
        # Get the name of the folder
        frun = self.x.GetFullFolderNames(i)
        # Get the number of iterations, etc.
        qtriq, ftriq, nStats, n0, nIter = self.GetTriqFile()
        # Save location
        fpwd = os.getcwd()
        # Enter the case folder
        os.chdir(frun)
        # Convert other format to TRIQ if necessary
        if qtriq:
            self.PreprocessTriq(ftriq, i=i)
        # Read the triangulation
        self.ReadTriq(ftriq)
        # Map the triangulation
        self.MapTriCompID()
        # Calculate the forces
        FM = self.GetTriqForces(i)
       # )
       # -----------------
       # Update Data Books
       # -----------------
       # (
        # Loop through patches
        for p in ([None] + self.patches):
            # Check if new case for this patch
            if j is None:
                # Increment the number of cases
                self[p].n += 1
                # Append trajectory values
                for k in self[p].xCols:
                    # Append to that column
                    self[p][k] = np.hstack((self[p][k], [self.x[k][i]]))
                # Append primary values
                for c in self[p].fCols:
                    # Get value
                    v = FM[p].get(c, np.nan)
                    # Save it.
                    self[p][c] = np.hstack((self[p][c], [v]))
                # Append iteration counts
                self[p]['nIter']  = np.hstack((self[p]['nIter'], [nIter]))
                self[p]['nStats'] = np.hstack((self[p]['nStats'], [nStats]))
            else:
                # Save updated trajectory values
                for k in self[p].xCols:
                    # Append to that column
                    self[p][k][j] = self.x[k][i]
                # Update data values
                for c in self[p].fCols:
                    # Save it.
                    self[p][c][j] = FM[p].get(c, np.nan)
                # Update the other statistics
                self[p]['nIter'][j]  = nIter
                self[p]['nStats'][j] = nStats
        # Write TRIQ/PLT/DAT file if requested
        self.WriteTriq(i, t=float(nIter))
        # Return to original folder
        os.chdir(fpwd)
       # )

  # >


# Data book target instance
class TargetDataBook(DataBookComp):
    r"""Class to handle data from data book target files

    There are more constraints on target files than the files that
    databook creates.

    :Call:
        >>> DBT = TargetDataBook(targ, x, opts, RootDir=None)
    :Inputs:
        *targ*: :class:`cape.cfdx.options.DataBook.TargetDataBook`
            Instance of a target source options interface
        *x*: :class:`pyCart.runmatrix.RunMatrix`
            Run matrix interface
        *opts*: :class:`cape.cfdx.options.Options`
            Options interface
        *RootDir*: :class:`str`
            Root directory, defaults to ``os.getcwd()``
    :Outputs:
        *DBT*: :class:`cape.cfdx.databook.TargetDataBook`
            Instance of the Cape data book target class
    :Versions:
        * 2015-01-10 ``@ddalle``: v1.0
        * 2015-12-14 ``@ddalle``: v1.1; add uncertainties
    """
  # ========
  # Config
  # ========
  # <
    # Initialization method
    def __init__(self, targ, x, opts, RootDir=None):
        """Initialization method

        :Versions:
            * 2014-12-21 ``@ddalle``: v1.0
            * 2015-06-03 ``@ddalle``: Added trajectory, split into methods
        """
        # Save the target options
        self.opts = opts
        self.topts = opts.get_TargetDataBookByName(targ)
        self.Name = targ
        # Save the trajectory.
        self.x = x.Copy()
        # Root directory
        if RootDir is None:
            # Default
            self.RootDir = os.getcwd()
        else:
            # Specified option
            self.RootDir = RootDir
        # Read the data
        self.ReadData()
        # Process the columns.
        self.ProcessColumns()
        # Make the trajectory data match the available list of points.
        self.UpdateRunMatrix()

    # Cannot use the dictionary disp on this; it's too huge
    def __repr__(self):
        r"""Representation method

        :Versions:
            * 2015-12-16 ``@ddalle``: v1.0
        """
        return "<TargetDataBook '%s', n=%i>" % (self.Name, self.n)
    __str__ = __repr__
  # >

  # ========
  # Readers
  # ========
  # <
    # Read the data
    def ReadData(self):
        r"""Read data file according to stored options

        :Call:
            >>> DBT.ReadData()
        :Inputs:
            *DBT*: :class:`cape.cfdx.databook.TargetDataBook`
                Instance of the data book target class
        :Versions:
            * 2015-06-03 ``@ddalle``: Copied from :func:`__init__` method
        """
        # Go to root directory
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Source file
        fname = self.topts.get_TargetFile()
        # Check for list
        if fname.__class__.__name__ == "list":
            # Join multiline string together
            fname = "".join(fname)
        # Check for the file.
        if not os.path.isfile(fname):
            raise IOError(
                "Target source file '%s' could not be found." % fname)
        # Delimiter
        delim = self.topts.get_DataBookDelimiter()
        # Comment character
        comchar = self.topts.get_CommentChar()
        # Open the file again.
        f = open(fname)
        # Loop until finding a line that doesn't begin with comment char.
        line = comchar
        nskip = -1
        while line.strip().startswith(comchar) or nskip < 1:
            # Save the old line.
            headers = line
            # Read the next line
            line = f.readline()
            nskip += 1
        # Close the file.
        f.close()
        # Translate into headers
        cols = headers.lstrip('#').strip().split(delim)
        # Strip
        self.headers = [col.strip() for col in cols]
        # Save number of points.
        self.nCol = len(self.headers)

        # Read it.
        try:
            # Read the target all at once.
            self.ReadAllData(fname, delimiter=delim, skiprows=nskip)
        except Exception:
            # Read the data by columns.
            self.ReadDataByColumn(fname, delimiter=delim, skiprows=nskip)
        # Go home
        os.chdir(fpwd)

    # Read the data file all at once.
    def ReadAllData(self, fname, delimiter=", ", skiprows=0):
        r"""Read target data file all at once

        :Call:
            >>> DBT.ReadAllData(fname, delimiter=", ", skiprows=0)
        :Inputs:
            *DBT*: :class:`cape.cfdx.databook.TargetDataBook`
                Instance of the Cape data book target class
            *fname*: :class:`str`
                Name of file to read
            *delimiter*: :class:`str`
                Data delimiter character(s)
            *skiprows*: :class:`int`
                Number of header rows to skip
        :Versions:
            * 2015-09-07 ``@ddalle``: v1.0
        """
        # Read the data.
        self.data = np.loadtxt(
            fname, delimiter=delimiter,
            skiprows=skiprows, dtype=float).transpose()
        # Save the number of cases.
        self.n = len(self.data[0])

    # Read data one column at a time
    def ReadDataByColumn(self, fname, delimiter=", ", skiprows=0):
        r"""Read target data one column at a time

        :Call:
            >>> DBT.ReadDataByColumn(fname, delimiter=", ", skiprows=0)
        :Inputs:
            *DBT*: :class:`cape.cfdx.databook.TargetDataBook`
                Instance of the Cape data book target class
            *fname*: :class:`str`
                Name of file to read
            *delimiter*: :class:`str`
                Data delimiter character(s)
            *skiprows*: :class:`int`
                Number of header rows to skip
        :Versions:
            * 2015-09-07 ``@ddalle``: v1.0
        """
        # Initialize data.
        self.data = []
        # Loop through columns.
        for i in range(self.nCol):
            # Try reading as a float second.
            try:
                self.data.append(
                    np.loadtxt(
                        fname, delimiter=delimiter,
                        skiprows=skiprows, dtype=float, usecols=(i,)))
                continue
            except Exception:
                pass
            # Try reading as a string last.
            self.data.append(
                np.loadtxt(
                    fname, delimiter=delimiter,
                    skiprows=skiprows, dtype="U", usecols=(i,)))
        # Number of cases
        self.n = len(self.data[0])

    # Read the columns and split into useful dict.
    def ProcessColumns(self):
        r"""Process data columns and split into dictionary keys

        :Call:
            >>> DBT.ProcessColumns()
        :Inputs:
            *DBT*: :class:`cape.cfdx.databook.TargetDataBook`
                Instance of the data book target class
        :Versions:
            * 2015-06-03 ``@ddalle``: Copied from :func:`__init__` method
            * 2015-12-14 ``@ddalle``: Added support for point sensors
        """
        # Initialize data fields.
        self.cols = []
        # Names of columns corresponding to trajectory keys.
        tkeys = self.topts.get_RunMatrix()
        # Loop through trajectory fields.
        for k in self.x.cols:
            # Get field name.
            col = tkeys.get(k, k)
            # Check for manually turned-off trajectory.
            if col is None:
                # Manually turned off.
                continue
            elif col not in self.headers:
                # Not present in the file.
                continue
            # Append the key.
            self.cols.append(col)
        # Initialize translations for force/moment coefficients
        ckeys = {}
        # List of potential components.
        tcomps = self.topts.get_TargetComponents()
        # Check for default.
        if tcomps is None:
            # Use all components.
            tcomps = self.opts.get_DataBookComponents()
        # Process the required fields.
        for comp in tcomps:
            # Initialize translations for this component.
            ckeys[comp] = {}
            # Get targets for this component.
            ctargs = self.opts.get_CompTargets(comp)
            # List of coefficients (i.e. no suffixes)
            coeffs = self.opts.get_DataBookCols(comp)
            # List of points or otherwise subcomponents
            pts = self.opts.get_DataBookPoints(comp)
            # Set default
            if pts is None or len(pts) == 0:
                pts = ['']
            # Loop through subcomponents (usually points or nothing)
            for pt in pts:
                # Loop through the possible coefficients
                for cf in coeffs:
                    # Loop through suffixes
                    for sfx in ['', 'std', 'min', 'max', 'uP', 'uM']:
                        # Get the field name and check its consistency
                        fi = self.CheckColumn(ctargs, pt, cf, sfx)
                        # Check for consistency/presence
                        if fi is None:
                            # Go to next line
                            continue
                        # Add the column.
                        self.cols.append(fi)
                        # Assemble coefficient/statistic name
                        c = '%s.%s_%s' % (pt, cf, sfx)
                        # Get rid of trivial point/suffix names
                        c = c.lstrip('/').lstrip('.').rstrip('_')
                        # Add to the translation dictionary.
                        ckeys[comp][c] = fi
        # Extract the data into a dict with a key for each relevant column.
        for col in self.cols:
            # Find it and save it as a key.
            self[col] = self.data[self.headers.index(col)]
        # Save the data keys translations.
        self.ckeys = ckeys

    # Check column presence and consistency
    def CheckColumn(self, ctargs, pt, cf, sfx):
        r"""Check a data book target column name and its consistency

        :Call:
            >>> fi = DBT.CheckColumn(ctargs, pt, c)
        :Inputs:
            *DBT*: :class:`cape.cfdx.databook.TargetDataBook`
                Instance of the data book target class
            *ctargs*: :class:`dict`
                Dictionary of target column names for each coefficient
            *pt*: :class:`str`
                Name of subcomponent (short for 'point')
            *c*: :class:`str`
                Name of the coefficient in question, including suffix
        :Outputs:
            *fi*: ``None`` | :class:`str`
                Name of the column in data book if present
        :Versions:
            * 2015-12-14 ``@ddalle``: v1.0
        """
        # Assemble coefficient/statistic name
        c = '%s.%s_%s' % (pt, cf, sfx)
        # Get rid of trivial point/suffix names
        c = c.lstrip('/').lstrip('.').rstrip('_')
        # Assemble default column name
        if pt and (cf.lower() == "cp") and ("Cp" not in self.headers):
            # Use the name of the point
            col = '%s_%s' % (pt, sfx)
        else:
            # Point.coeff_sfx
            col = '%s_%s' % (cf, sfx)
        # Get rid of trivial suffix names
        col = col.rstrip('_')
        # Get the translated name
        ctarg = ctargs.get(c, col)
        # Ensure list
        if ctarg.__class__.__name__ != "list":
            # Make it a list
            ctarg = [ctarg]
        # Loop through candidate targets
        for ct in ctarg:
            # Get the target source for this entry.
            if '/' not in ct:
                # Only one target source; assume it's this one.
                ti = self.Name
                fi = ct
            else:
                # Name of target/Name of column
                ti = ct.split('/')[0]
                fi = '/'.join(ct.split('/')[1:])
            # Check if the target is from this target source.
            if ti != self.Name:
                continue
            # Check if the column is present in the headers.
            if fi not in self.headers:
                # Check for default.
                if ct in ctargs:
                    # Manually specified and not recognized: error
                    raise KeyError(
                        "Missing data book target field:" +
                        " TargetDataBook='%s', " % self.Name +
                        " ctarg='%s', " % ct +
                        " coeff='%s', " % c +
                        " column='%s', " % fi)
                else:
                    # Autoselected name but not in the file.
                    continue
            # Return the column name
            return fi
  # >

  # ======
  # Data
  # ======
  # <
    # Get a value
    def GetCoeff(self, comp, coeff, I, **kw):
        r"""Get a coefficient value for one or more cases

        :Call:
            >>> v = DBT.GetCoeff(comp, coeff, i)
            >>> V = DBT.GetCoeff(comp, coeff, I)
        :Inputs:
            *DBT*: :class:`cape.cfdx.databook.TargetDataBook`
                Instance of the Cape data book target class
            *comp*: :class:`str`
                Component whose coefficient is being plotted
            *coeff*: :class:`str`
                Coefficient being plotted
            *i*: :class:`int`
                Individual case/entry index
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indexes of cases to include in sweep
        :Outputs:
            *v*: :class:`float`
                Scalar value from the appropriate column
            *V*: :class:`np..ndarray`
                Array of values from the appropriate column
        :Versions:
            * 2018-02-12 ``@ddalle``: v1.0
        """
        # Check for patch delimiter
        if "/" in comp:
            # Format: Cp_ports.P001
            compo, pt = comp.split("/")
        elif "." in comp:
            # Format: Cp_ports/P001
            compo, pt = comp.split(".")
        else:
            # Only comp given; use total of point names
            compo = comp
            pt = None
        # List of keys available for this component
        ckeys = self.ckeys.get(compo, {})
        # Get point if applicable
        if pt is not None:
            # Add point/patch/whatever name
            ccoeff = "%s.%s" % (pt, coeff)
        else:
            # Use name of coefficient directly
            ccoeff = coeff
        # Get the key
        ckey = ckeys.get(ccoeff, coeff)
        # Check validity
        if ckey not in self:
            raise KeyError(
                "No key '%s' for component '%s', coefficient '%s'"
                % (ckey, comp, coeff))
        # Get the value
        return self[ckey][I]

  # >

  # =============
  # Organization
  # =============
  # <
    # Match the databook copy of the trajectory
    def UpdateRunMatrix(self):
        """Match the trajectory to the cases in the data book

        :Call:
            >>> DBT.UpdateRunMatrix()
        :Inputs:
            *DBT*: :class:`cape.cfdx.databook.TargetDataBook`
                Instance of the data book target class
        :Versions:
            * 2015-06-03 ``@ddalle``: v1.0
        """
        # Get trajectory key specifications.
        tkeys = self.topts.get_RunMatrix()
        # Loop through the trajectory keys.
        for k in self.x.cols:
            # Get the column name in the target.
            tk = tkeys.get(k, k)
            # Set the value if it's a default.
            tkeys.setdefault(k, tk)
            # Check for ``None``
            if (tk is None) or (tk not in self):
                # Use NaN as the value.
                self.x[k] = np.nan*np.ones(self.n)
                # Set the value.
                tkeys[k] = None
                continue
            # Update the trajectory values to match those of the trajectory.
            self.x[k] = self[tk]
            # Set the text.
            self.x.text[k] = [str(xk) for xk in self[tk]]
        # Save the key translations.
        self.xkeys = tkeys
        # Set the number of cases in the "trajectory."
        self.x.nCase = self.n

    # Find an entry by trajectory variables.
    def FindMatch(self, DBc, i):
        r"""Find an entry by run matrix (trajectory) variables

        Cases will be considered matches by comparing variables specified in
        the *DataBook* section of :file:`cape.json` as cases to compare
        against.  Suppose that the control file contains the following.

        .. code-block:: javascript

            "DataBook": {
                "Targets": {
                    "Experiment": {
                        "File": "WT.dat",
                        "RunMatrix": {"alpha": "ALPHA", "Mach": "MACH"}
                        "Tolerances": {
                            "alpha": 0.05,
                            "Mach": 0.01
                        }
                    }
                }
            }

        Then any entry in the data book target that matches the Mach number
        within 0.01 (using a column labeled *MACH*) and alpha to within 0.05 is
        considered a match.  If there are more trajectory variables, they are
        not used for this filtering of matches.

        :Call:
            >>> j = DBT.FindMatch(x, i)
        :Inputs:
            *DBT*: :class:`cape.cfdx.databook.TargetDataBook`
                Instance of the Cape data book target data carrier
            *x*: :class:`cape.runmatrix.RunMatrix`
                The current pyCart trajectory (i.e. run matrix)
            *i*: :class:`int`
                Index of the case from the trajectory to try match
        :Outputs:
            *j*: :class:`numpy.ndarray`\ [:class:`int`]
                Array of indices that match the trajectory within tolerances
        :See also:
            * :func:`cape.cfdx.databook.DataBookComp.FindTargetMatch`
            * :func:`cape.cfdx.databook.DataBookComp.FindMatch`
        :Versions:
            * 2014-12-21 ``@ddalle``: v1.0
            * 2016-06-27 ``@ddalle``: v1.1; move to ``DataBookComp``
            * 2018-02-12 ``@ddalle``: v.12; First arg ``DataBookComp``
        """
        # Use the target-oriented method
        return self.FindTargetMatch(DBc, i, self.topts, keylist='tol')
  # >

  # ======
  # Plot
  # ======
  # <
    # Plot a sweep of one or more coefficients
    def PlotCoeff(self, comp, coeff, I, **kw):
        r"""Plot a sweep of one coefficient over several cases

        :Call:
            >>> h = DBT.PlotCoeff(comp, coeff, I, **kw)
        :Inputs:
            *DBT*: :class:`cape.cfdx.databook.TargetDataBook`
                Instance of the Cape data book target class
            *comp*: :class:`str`
                Component whose coefficient is being plotted
            *coeff*: :class:`str`
                Coefficient being plotted
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indexes of cases to include in sweep
        :Keyword Arguments:
            *x*: [ {None} | :class:`str` ]
                RunMatrix key for *x* axis (or plot against index if ``None``)
            *Label*: [ {*comp*} | :class:`str` ]
                Manually specified label
            *Legend*: [ {True} | False ]
                Whether or not to use a legend
            *StDev*: [ {None} | :class:`float` ]
                Multiple of iterative history standard deviation to plot
            *MinMax*: [ {False} | True ]
                Whether to plot minimum and maximum over iterative history
            *Uncertainty*: [ {False} | True ]
                Whether to plot direct uncertainty
            *PlotOptions*: :class:`dict`
                Plot options for the primary line(s)
            *StDevOptions*: :class:`dict`
                Dictionary of plot options for the standard deviation plot
            *MinMaxOptions*: :class:`dict`
                Dictionary of plot options for the min/max plot
            *UncertaintyOptions*: :class:`dict`
                Dictionary of plot options for the uncertainty plot
            *FigureWidth*: :class:`float`
                Width of figure in inches
            *FigureHeight*: :class:`float`
                Height of figure in inches
            *PlotTypeStDev*: [ {'FillBetween'} | 'ErrorBar' ]
                Plot function to use for standard deviation plot
            *PlotTypeMinMax*: [ {'FillBetween'} | 'ErrorBar' ]
                Plot function to use for min/max plot
            *PlotTypeUncertainty*: [ 'FillBetween' | {'ErrorBar'} ]
                Plot function to use for uncertainty plot
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :Versions:
            * 2015-05-30 ``@ddalle``: v1.0
            * 2015-12-14 ``@ddalle``: Added uncertainties
        """
        # Check for patch delimiter
        if "/" in comp:
            # Format: Cp_ports.P001
            compo, pt = comp.split("/")
        elif "." in comp:
            # Format: Cp_ports/P001
            compo, pt = comp.split(".")
        else:
            # Only comp given; use total of point names
            compo = comp
            pt = None
        # List of keys available for this component
        ckeys = self.ckeys.get(compo)
        # Check availability
        if (ckeys is None) or (coeff not in ckeys):
            # Check for special cases
            if coeff in ['cp', 'CP']:
                # Special case; try to plot anyway
                pass
            else:
                # Key not available
                return
        # Get point if applicable
        if pt is not None:
            # Add point/patch/whatever name
            ccoeff = "%s.%s" % (pt, coeff)
        else:
            # Use name of coefficient directly
            ccoeff = coeff
        # Get the key
        ckey = ckeys.get(ccoeff, coeff)
        # Get horizontal key.
        xk = kw.get('x')
        # Process this key to turn it into a trajectory column
        if xk is None or xk == 'Index':
            # This is fine
            pass
        elif xk in self.xkeys:
            # Set the key to the translated value (which may be the same).
            kw['x'] = self.xkeys[xk]
        elif xk in [
            "alpha", "alpha_m", "aoam",
            "phi_m", "phim", "beta", "phi"
        ]:
            # Special allowed keys
            pass
        else:
            # No translation for this key
            raise ValueError(
                "No trajectory key translation known for key '%s'" % xk)
        # Flip the error bar default plot types
        kw.setdefault('PlotTypeMinMax',      'ErrorBar')
        kw.setdefault('PlotTypeUncertainty', 'FillBetween')
        kw.setdefault('PlotTypeStDev',       'ErrorBar')
        # Prep keyword inputs for default settings
        kw.setdefault('PlotOptions', {})
        # Alter the default settings for the line
        kw['PlotOptions'].setdefault('color', 'r')
        kw['PlotOptions'].setdefault('zorder', 7)
        # Save the component name
        kw['comp'] = comp
        # Call the base plot method
        return self.PlotCoeffBase(ckey, I, **kw)
  # >


# Data book for an individual component
class TimeSeriesDataBook(DataBookComp):
    r"""Individual force & moment component data book

    This class is derived from :class:`DataBookComp`.

    :Call:
        >>> DBi = TimeSeriesDataBook(comp, cntl, **kw)
    :Inputs:
        *comp*: :class:`str`
            Name of the component
        *cntl*: :class:`Cntl`
            CAPE control class instance
        *targ*: {``None``} | :class:`str`
            If used, read a duplicate data book as a target named *targ*
        *check*: ``True`` | {``False``}
            Whether or not to check LOCK status
        *lock*: ``True`` | {``False``}
            If ``True``, wait if the LOCK file exists
    :Outputs:
        *DBi*: :class:`cape.cfdx.databook.FMDataBook`
            An individual component data book
    :Versions:
        * 2024-10-09 ``@aburkhea``: Started
    """
  # ========
  # Config
  # ========
  # <
    # Initialization method
    def __init__(self, comp, cntl, targ=None, check=False, lock=False, **kw):
        """Initialization method

        :Versions:
            * 2024-10-09 ``@aburkhea``: Started
        """
        # Unpack *cntl*
        x = cntl.x
        opts = cntl.opts
        # Save relevant inputs
        self.x = x
        self.opts = opts
        self.cntl = cntl
        self.comp = comp
        self.name = comp
        self.sources = {}
        # Root directory
        self.RootDir = kw.get("RootDir", os.getcwd())

        # Get the directory.
        if targ is None:
            # Primary data book directory
            fdir = opts.get_DataBookFolder()
        else:
            # Secondary data book directory
            fdir = opts.get_TargetDataBookDir(targ)

        # Construct the file name.
        fcomp = 'ts_%s.csv' % comp
        # Folder name for compatibility.
        fdir = fdir.replace("/", os.sep)
        fdir = fdir.replace("\\", os.sep)
        # Construct the full file name.
        fname = os.path.join(fdir, fcomp)
        # Save the file name.
        self.fname = fname
        self.fdir = fdir

        # Safely change to root directory
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Create directories if necessary
        if not os.path.isdir(fdir):
            # Create data book folder (should not occur)
            os.mkdir(fdir)
        # Check for lineload folder
        if not os.path.isdir(os.path.join(fdir, 'timeseries')):
            # Create line load folder
            os.mkdir(os.path.join(fdir, 'timeseries'))
        # Return to original location
        os.chdir(fpwd)

        # Process columns
        self.ProcessColumns()

        # Read the file or initialize empty arrays.
        self.Read(self.fname, check=check, lock=lock)

        # Save the target translations
        self.targs = opts.get_CompTargets(comp)
        # Divide columns into parts
        self.DataCols = opts.get_DataBookDataCols(comp)

    # Command-line representation
    def __repr__(self):
        r"""Representation method

        :Versions:
            * 2024-10-09 ``@aburkhea``: v1.0
        """
        # Initialize string
        lbl = "<TimeSeriesDataBook %s, " % self.comp
        # Add the number of conditions.
        lbl += "nCase=%i>" % self.n
        # Output
        return lbl
    # String conversion
    __str__ = __repr__
  # >
  # >

  # ======
  # Read
  # ======
  # <

    def ReadCase(self, comp):
        r"""Read a :class:`CaseTS` object

        :Call:
            >>> fm = DB.ReadCase(comp)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *fm*: :class:`cape.cfdx.databook.CaseFM`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseTS(comp)

    # Read residual history class
    def ReadCaseResid(self):
        r"""Read a :class:`CaseResid` object

        :Call:
            >>> H = DB.ReadCaseResid()
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBookComp`
                Instance of data book class
        :Outputs:
            *H*: :class:`cape.cfdx.databook.CaseResid`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseResid()

    # Read a copy
    def ReadCopy(self, check=False, lock=False):
        r"""Read a copied database object

        :Call:
            >>> DBc1 = DBc.ReadCopy(check=False, lock=False)
        :Inputs:
            *DBc*: :class:`cape.cfdx.databook.DataBookComp`
                Data book base object
            *check*: ``True`` | {``False``}
                Whether or not to check LOCK status
            *lock*: ``True`` | {``False``}
                If ``True``, wait if the LOCK file exists
        :Outputs:
            *DBc1*: :class:`cape.cfdx.databook.DataBookComp`
                Copy of data book base object
        :Versions:
            * 2017-06-26 ``@ddalle``: v1.0
        """
        # Check for a name
        try:
            # Use the *name* as the first choice
            name = self.name
        except AttributeError:
            # Fall back to the *comp* attribute
            name = self.comp
        # Call the object
        DBc = self.__class__(name, self.cntl, check=check, lock=lock)
        # Ensure the same root directory is used
        DBc.RootDir = getattr(self, "RootDir", os.getcwd())
        # Output
        return DBc
  # >
  # ======
  # Write
  # ======
  # <

    # Delete Time Series Case
    def DeleteCases(self, I, comp):
        # Default case list
        if I is None:
            # Use all trajectory points
            I = range(self.x.nCase)
        # Check type
        if self.opts.get_DataBookType(comp) != "TimeSeries":
            raise ValueError(
                "Component '%s' is not a TimeSeries component" % comp)
        # Get the data book
        DBc = self
        # Number of cases in current data book.
        nCase = DBc.n
        # Initialize data book index array.
        J = []
        # Loop though indices to delete.
        for i in I:
            # Find the match.
            j = DBc.FindMatch(i)
            # Check if one was found.
            if j is None:
                continue
            # Append to the list of data book indices.
            J.append(j)
        # Number of deletions
        nj = len(J)
        # Exit if no deletions
        if nj == 0:
            return 0
        # Initialize mask of cases to keep.
        mask = np.ones(nCase, dtype=bool)
        # Set values equal to false for cases to be deleted.
        mask[J] = False
        # Loop through data book columns.
        for c in DBc.keys():
            # Apply the mask
            DBc[c] = DBc[c][mask]
        # Update the number of entries.
        DBc.n = len(DBc[list(DBc.keys())[0]])
        # Also remove the time series cdb
        # Get the name of the folder
        frun = self.x.GetFullFolderNames(i)
        # Build names for cdb file
        fts  = os.path.join(self.RootDir, DBc.fdir, 'timeseries')
        fcas = os.path.join(fts, frun)
        # CAPE db file name
        fcdb = os.path.join(fcas, '%s.cdb' % (comp))
        # Remove fcdb if it exists
        if os.path.isfile(fcdb):
            os.remove(fcdb)
        # Output
        return nj

    # Update or add an entry for one component
    def UpdateCaseDB(self, i, j, comp):
        r"""Update or add a case to a data book

        The history of a run directory is processed if either one of
        three criteria are met.

            1. The case is not already in the data book
            2. The most recent iteration is greater than the data book
               value
            3. The number of iterations used to create statistics has
               changed

        :Call:
            >>> n = DB.UpdateCaseComp(i, comp)
        :Inputs:
            *DB*: :class:`pyFun.databook.DataBook`
                Instance of the data book class
            *i*: :class:`int`
                RunMatrix index
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *n*: ``0`` | ``1``
                How many updates were made
        :Versions:
            * 2014-12-22 ``@ddalle``: v1.0
            * 2017-04-12 ``@ddalle``: Modified to work one component
            * 2017-04-23 ``@ddalle``: Added output
        """
        # Get the first data book component.
        DBc = self
        # Get the current iteration number
        nIter = self.cntl.GetCurrentIter(i)
        # Get the number of iterations used for statutils.
        nStats = self.opts.get_DataBookNStats(comp)
        # Get the iteration at which statistics can begin.
        nMin = self.opts.get_DataBookNMin(comp)
        # Maximum number of iterations allowed
        nMaxStats = self.opts.get_DataBookNMaxStats(comp)
        # Limit max stats if instructed to do so
        if nMaxStats is None:
            # No max
            nMax = None
        else:
            # Specified max, but don't use data before *nMin*
            nMax = min(nIter - nMin, nMaxStats)
        # Get the name of the folder.
        frun = self.x.GetFullFolderNames(i)
        # Status update.
        print(frun)
        # Go home.
        os.chdir(self.RootDir)
        # Check if the folder exists.
        if not os.path.isdir(frun):
            # Nothing to do.
            return 0
        # Go to the folder.
        os.chdir(frun)
        # Read residual
        H = self.ReadCaseResid()
       # --- Read Iterative History ---
        # Get component (note this automatically defaults to *comp*)
        compID = self.opts.get_DataBookCompID(comp)
        # Check for multiple components
        if type(compID).__name__ in ['list', 'ndarray']:
            # Read the first component
            FM = self.ReadCase(compID[0])
            # Loop through remaining components
            for compi in compID[1:]:
                # Check for minus sign
                if compi.startswith('-'):
                    # Subtract the component
                    FM -= self.ReadCase(compi.lstrip('-'))
                else:
                    # Add in the component
                    FM += self.ReadCase(compi)
        else:
            # Read the iterative history for single component
            FM = self.ReadCase(compID)
        # List of transformations
        tcomp = self.opts.get_DataBookTransformations(comp)
        tcomp = list(tcomp)
        # Special transformation to reverse *CLL* and *CLN*
        tflight = {
            "Type": "ScaleCoeffs",
            "CLL": -1.0,
            "CLN": -1.0
        }
        # Check for ScaleCoeffs
        for tj in tcomp:
            # Skip if not a "ScaleCoeffs"
            if tj.get("Type") != "ScaleCoeffs":
                continue
            # Use it if we have either *CLL* or *CLM*
            if "CLL" in tj or "CLN" in tj:
                break
        else:
            # If we didn't find a match, append *tflight*
            tcomp.append(tflight)
        # Save the Lref, current MRP to any "ShiftMRP" transformations
        for topts in tcomp:
            # Get type
            ttyp = topts.get("Type")
            # Only apply to "ShiftMRP"
            if ttyp == "ShiftMRP":
                # Use a copy to avoid changing cntl.opts
                topts = dict(topts)
                # Component to use for current MRP
                compID = self.cntl.opts.get_DataBookCompID(comp)
                if isinstance(compID, list):
                    compID = compID[0]
                # Reset points for default *FromMRP*
                self.cntl.opts.reset_Points()
                # Use MRP prior to transfformations as default *FromMRP*
                x0 = self.cntl.opts.get_RefPoint(comp)
                # Ensure points are calculated
                self.cntl.PreparePoints(i)
                # Use post-transformation MRP as default *ToMRP*
                x1 = self.cntl.opts.get_RefPoint(comp)
                # Get current Lref
                Lref = self.cntl.opts.get_RefLength(comp)
                # Set those as defaults in transformation
                x0 = topts.setdefault("FromMRP", x0)
                x1 = topts.setdefault("ToMRP", x1)
                topts.setdefault("RefLength", Lref)
                # Expand if *x0* is a string
                topts["FromMRP"] = self.cntl.opts.expand_Point(x0)
                topts["ToMRP"] = self.cntl.opts.expand_Point(x1)
            # Apply the transformation.
            FM.TransformFM(topts, self.x, i)
        # Process the statistics.
        s = FM.GetStats(nStats, nMax)
        # Get the corresponding residual drop
        if 'nOrders' in DBc:
            nOrders = H.GetNOrders(s['nStats'])
        # Write case time series cdbs
        fts  = os.path.join(self.RootDir, self.fdir, 'timeseries')
        fgrp = os.path.join(fts, frun.split(os.sep)[0])
        fcas = os.path.join(fts, frun)
        # Create folders as necessary
        for f1 in (fts, fgrp, fcas):
            if not os.path.isdir(f1):
                os.mkdir(f1)
        # CAPE db file name
        fcdb = os.path.join(fcas, '%s.cdb' % (comp))
        # Only write minimal cols to minimize data duplication
        cols0 = FM.cols[:]
        for col in cols0:
            # Burst cols not in base_cols
            if col not in FM._base_cols:
                _ = FM.burst_col(col)
        FM.write_dbook_cdb(fname=fcdb)
        # Get end time
        tEnd = FM.get_tend(CASE_COL_TIME)
        # Save the data.
        if j is None:
            # Add to the number of cases.
            DBc.n += 1
            # Append trajectory values.
            for k in self.x.cols:
                # Append
                DBc[k] = np.append(DBc[k], self.x[k][i])
            # Append values.
            for c in DBc.DataCols:
                if c in s:
                    DBc[c] = np.append(DBc[c], s[c])
                else:
                    DBc[c] = np.append(DBc[c], np.nan)
            # Append residual drop.
            if 'nOrders' in DBc:
                DBc['nOrders'] = np.hstack((DBc['nOrders'], [nOrders]))
            # Append iteration counts.
            if 'nIter' in DBc:
                DBc['nIter']  = np.hstack((DBc['nIter'], [nIter]))
            if 'nStats' in DBc:
                DBc['nStats'] = np.hstack((DBc['nStats'], [s['nStats']]))
            # Append end time.
            if 'tEnd' in DBc:
                DBc['tEnd']  = np.hstack((DBc['tEnd'], [tEnd]))

        else:
            # Save updated trajectory values
            for k in DBc.xCols:
                # Append to that column
                DBc[k][j] = self.x[k][i]
            # Update data values.
            for c in DBc.DataCols:
                DBc[c][j] = s[c]
            # Update the other statistics.
            if 'nOrders' in DBc:
                DBc['nOrders'][j] = nOrders
            if 'nIter' in DBc:
                DBc['nIter'][j]   = nIter
            if 'nStats' in DBc:
                DBc['nStats'][j]  = s['nStats']
            if 'tEnd' in DBc:
                DBc['tEnd'][j]    = tEnd
        # Go back.
        os.chdir(self.RootDir)
  # >


# Aerodynamic history class
class DataBook(DataBookBase):
    r"""Interface to the data book for a given CFD run matrix

    :Call:
        >>> DB = cape.cfdx.databook.DataBook(cntl, **kw)
    :Inputs:
        *cntl*: :class:`Cntl`
            CAPE control class instance
        *RootDir*: :class:`str`
            Root directory, defaults to ``os.getcwd()``
        *targ*: {``None``} | :class:`str`
            Option to read duplicate data book as a target named *targ*
    :Outputs:
        *DB*: :class:`cape.cfdx.databook.DataBook`
            Instance of the Cape data book class
        *DB.x*: :class:`cape.runmatrix.RunMatrix`
            Run matrix of rows saved in the data book
        *DB[comp]*: :class:`cape.cfdx.databook.FMDataBook`
            Component data book for component *comp*
        *DB.Components*: :class:`list`\ [:class:`str`]
            List of force/moment components
        *DB.Targets*: :class:`dict`
            Dictionary of :class:`TargetDataBook` target data books
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
        * 2015-01-10 ``@ddalle``: v1.0
        * 2022-03-07 ``@ddalle``: v1.1; allow .cntl
    """
    _fm_cls = FMDataBook
    _ts_cls = TimeSeriesDataBook
    _prop_cls = PropDataBook
    _pyfunc_cls = PyFuncDataBook
  # ======
  # Config
  # ======
  # <

    # Initialization method
    def __init__(
            self,
            cntl,
            RootDir: Optional[str] = None,
            targ: Optional[str] = None, **kw):
        r"""Initialization method

        :Versions:
            * 2014-12-21 ``@ddalle``: v1.0
        """
        # Root directory
        if RootDir is None:
            # Default
            self.RootDir = os.getcwd()
        else:
            # Specified option
            self.RootDir = RootDir
        # Unpack options and run matrix
        x = cntl.x
        opts = cntl.opts
        # Save control instance (recursive, but that's ok)
        self.cntl = cntl
        # Change safely to the root folder
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Lock status
        check = kw.get("check", False)
        lock  = kw.get("lock",  False)
        # Get list of components
        comp = kw.get('comp')
        # Default list of components
        if comp is None:
            # Default: all components
            comps = opts.get_DataBookComponents(targ=targ)
        elif type(comp).__name__ in ['str', 'unicode']:
            # Split by comma (also ensures list)
            comps = comp.split(',')
        else:
            # Already a list?
            comps = comp
        # Save the components
        self.Components = comps
        # Save the folder
        if targ is None:
            # Root data book
            self.Dir = opts.get_DataBookFolder()
        else:
            # Read data book as a target that duplicates the root
            self.Dir = opts.get_TargetDataBookDir(targ)
            # Save target options
            self.topts = opts.get_TargetDataBookByName(targ)
        # Save the trajectory.
        self.x = x.Copy()
        # Save the options.
        self.opts = opts
        self.targ = targ
        # Go to root if necessary
        if os.path.isabs(self.Dir):
            os.chdir("/")
        # Make sure the destination folder exists.
        for fdir in self.Dir.split('/'):
            # If folder ends in os.sep; go on
            if not fdir:
                continue
            # Check if the folder exists.
            if not os.path.isdir(fdir):
                os.mkdir(fdir)
            # Go to the folder.
            os.chdir(fdir)
        # Go back to root folder.
        os.chdir(self.RootDir)
        # Loop through the components.
        for comp in comps:
            # Get component type
            tcomp = opts.get_DataBookType(comp)
            # Get handle to reader
            rdrfunc = self._readers.get(tcomp)
            # Initialize the data book.
            rdrfunc(self, comp, check=check, lock=lock)
            # # Initialize the data book.
        # Initialize targets.
        self.Targets = {}
        # Return to original location
        os.chdir(fpwd)

    # Command-line representation
    def __repr__(self):
        r"""Representation method

        :Versions:
            * 2014-12-22 ``@ddalle``: v1.0
        """
        # Get class
        cls = self.__class__
        clsname = cls.__name__
        # Get module
        modname = cls.__module__
        # Get base module
        modbase = modname.split('.')[1]
        # Initialize String
        lbl = f"<{modbase}.{clsname}, "
        # Add the number of components.
        lbl += "ncomp=%i>" % len(self.Components)
        # Output
        return lbl
    # String conversion
    __str__ = __repr__

    # Directory creation using appropriate settings
    def mkdir(self, fdir):
        r"""Create a directory using settings from *DataBook>umask*

        :Call:
            >>> DB.mkdir(fdir)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the Cape data book class
            *fdir*: :class:`str`
                Directory to create
        :Versions:
            * 2017-09-05 ``@ddalle``: v1.0
        """
        # Call databook method
        os.mkdir(fdir)
  # >

  # ===
  # I/O
  # ===
  # <
    # Write the data book
    def Write(self, unlock=True):
        r"""Write the current data book in Python memory to file

        :Call:
            >>> DB.Write(unlock=True)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the Cape data book class
        :Versions:
            * 2014-12-22 ``@ddalle``: v1.0
            * 2015-06-19 ``@ddalle``: New multi-key sort
            * 2017-06-12 ``@ddalle``: Added *unlock*
        """
        # Start from root directory.
        os.chdir(self.RootDir)
        # Get the sort key.
        skey = self.opts.get_SortKey()
        # Sort the data book if there is a key.
        if skey is not None:
            # Sort on either a single key or multiple keys.
            self.Sort(skey)
        # Loop through the components.
        for comp in self.Components:
            # Check the component type.
            tcomp = self.opts.get_DataBookType(comp)
            if tcomp not in ['Force', 'Moment', 'FM']:
                continue
            # Write individual component.
            self[comp].Write(unlock=unlock)

    # Initialize a FMDataBook object
    def ReadFM(self, comp, check=False, lock=False):
        r"""Initialize data book for one component

        :Call:
            >>> DB.InitFMDataBook(comp, check=False, lock=False)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the pyCart data book class
            *comp*: :class:`str`
                Name of component
            *check*: ``True`` | {``False``}
                Whether or not to check for LOCK file
            *lock*: ``True`` | {``False``}
                Whether or not to create LOCK file
        :Versions:
            * 2015-11-10 ``@ddalle``: v1.0
            * 2017-04-13 ``@ddalle``: Self-contained and renamed
        """
        self[comp] = self._fm_cls(
            comp, self.cntl,
            targ=self.targ, check=check, lock=lock, RootDir=self.RootDir)

    # Initialize a FMDataBook object
    def ReadDBCompTS(self, comp, check=False, lock=False):
        r"""Initialize time series data book for one component

        :Call:
            >>> DB.InitDBComp(comp, check=False, lock=False)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the pyCart data book class
            *comp*: :class:`str`
                Name of component
            *check*: ``True`` | {``False``}
                Whether or not to check for LOCK file
            *lock*: ``True`` | {``False``}
                Whether or not to create LOCK file
        :Versions:
            * 2015-11-10 ``@ddalle``: v1.0
            * 2017-04-13 ``@ddalle``: Self-contained and renamed
        """
        self[comp] = self._ts_cls(
            comp, self.cntl,
            targ=self.targ, check=check, lock=lock, RootDir=self.RootDir)

    # Initialize a FMDataBook object
    def ReadDBCaseProp(self, comp, check=False, lock=False):
        r"""Initialize data book for one component

        :Call:
            >>> DB.InitDBComp(comp, check=False, lock=False)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the pyCart data book class
            *comp*: :class:`str`
                Name of component
            *check*: ``True`` | {``False``}
                Whether or not to check for LOCK file
            *lock*: ``True`` | {``False``}
                Whether or not to create LOCK file
        :Versions:
            * 2015-11-10 ``@ddalle``: v1.0
            * 2017-04-13 ``@ddalle``: Self-contained and renamed
        """
        self[comp] = self._prop_cls(
            comp, self.cntl,
            targ=self.targ, check=check, lock=lock, RootDir=self.RootDir)

    # Initialize a FMDataBook object
    def ReadPyFuncDataBook(self, comp, check=False, lock=False):
        r"""Initialize data book for one PyFunc component

        :Call:
            >>> DB.ReadPyFuncDataBook(comp, check=False, lock=False)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the pyCart data book class
            *comp*: :class:`str`
                Name of component
            *check*: ``True`` | {``False``}
                Whether or not to check for LOCK file
            *lock*: ``True`` | {``False``}
                Whether or not to create LOCK file
        :Versions:
            * 2022-04-10 ``@ddalle``: v1.0
        """
        # Read databook component
        self[comp] = self._pyfunc_cls(
            comp, self.cntl,
            targ=self.targ, check=check, lock=lock)

    # Read line load
    def ReadLineLoad(self, comp, conf=None, targ=None, **kw):
        r"""Read a line load data

        :Call:
            >>> DB.ReadLineLoad(comp)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the pycart data book class
            *comp*: :class:`str`
                Line load component group
            *conf*: {``None``} | :class:`cape.config.Config`
                Surface configuration interface
            *targ*: {``None``} | :class:`str`
                Alternate directory to read from, else *DB.targ*
        :Versions:
            * 2015-09-16 ``@ddalle``: v1.0
            * 2016-06-27 ``@ddalle``: Added *targ*
        """
        # Initialize if necessary
        try:
            self.LineLoads
        except AttributeError:
            self.LineLoads = {}
        # Try to access the line load
        try:
            if targ is None:
                # Check for the line load data book as is
                self.LineLoads[comp]
            else:
                # Check for the target
                self.ReadTarget(targ)
                # Check for the target line load
                self.Targets[targ].LineLoads[comp]
        except Exception:
            # Safely go to root directory
            fpwd = os.getcwd()
            os.chdir(self.RootDir)
            # Read the target
            self._LineLoadDataBook(comp, conf=conf, targ=targ)
            # Return to starting location
            os.chdir(fpwd)

    # Local line load data book read
    def _LineLoadDataBook(self, comp, conf=None, targ=None):
        r"""Versions-specific line load reader

        :Versions:
            * 2017-04-18 ``@ddalle``: v1.0
        """
        pass

    # Read TriqPoint components
    def _ReadTriqPoint(self, comp, check=False, lock=False, **kw):
        self.ReadTriqPoint(comp, check=check, lock=lock, **kw)

    # Find first force/moment component
    def GetRefComponent(self):
        r"""Get first component with type 'FM', 'Force', or 'Moment'

        :Call:
            >>> DBc = DB.GetRefComponent()
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Data book instance
        :Outputs:
            *DBc*: :class:`cape.cfdx.databook.FMDataBook`
                Data book for one component
        :Versions:
            * 2016-08-18 ``@ddalle``: v1.0
        """
        # Loop through components
        for comp in self.Components:
            # Get the component type
            typ = self.opts.get_DataBookType(comp)
            # Check if it's in the desirable range
            if typ in ['FM', 'Force', 'Moment']:
                # Use this component
                return self[comp]

    # Function to read targets if necessary
    def ReadTarget(self, targ):
        r"""Read a data book target if it is not already present

        :Call:
            >>> DB.ReadTarget(targ)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the Cape data book class
            *targ*: :class:`str`
                Target name
        :Versions:
            * 2015-09-16 ``@ddalle``: v1.0
        """
        # Initialize targets if necessary
        try:
            self.Targets
        except AttributeError:
            self.Targets = {}
        # Try to access the target.
        try:
            self.Targets[targ]
        except Exception:
            # Get the target type
            typ = self.opts.get_TargetDataBookType(targ).lower()
            # Check the type
            if typ in ['duplicate', 'cape', 'pycart', 'pyfun', 'pyover']:
                # Read a duplicate data book
                self._DataBook(targ)
                # Update the trajectory
                self.Targets[targ].UpdateRunMatrix()
            else:
                # Read the file.
                self._TargetDataBook(targ)

    # Read TriqFM components
    def ReadTriqFM(self, comp, check=False, lock=False):
        r"""Read a TriqFM data book if not already present

        :Call:
            >>> DB.ReadTriqFM(comp, check=False, lock=False)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Data book instance
            *comp*: :class:`str`
                Name of TriqFM component
            *check*: ``True`` | {``False``}
                Whether or not to check LOCK status
            *lock*: ``True`` | {``False``}
                If ``True``, wait if the LOCK file exists
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
        """
        # Initialize if necessary
        try:
            self.TriqFM
        except Exception:
            self.TriqFM = {}
        # Try to access the TriqFM database
        try:
            self.TriqFM[comp]
            # Confirm lock
            if lock:
                self.TriqFM[comp].Lock()
        except Exception:
            # Safely go to root directory
            fpwd = os.getcwd()
            os.chdir(self.RootDir)
            # Read data book
            self.TriqFM[comp] = TriqFMDataBook(
                self.x, self.opts, comp,
                RootDir=self.RootDir, check=check, lock=lock)
            # Return to starting position
            os.chdir(fpwd)

    # Local version of data book
    def _DataBook(self, targ):
        self.Targets[targ] = DataBook(
            self.x, self.opts, RootDir=self.RootDir, targ=targ)

    # Local version of target
    def _TargetDataBook(self, targ):
        self.Targets[targ] = TargetDataBook(
            targ, self.x, self.opts, self.RootDir)
  # >

  # ========
  # Case I/O
  # ========
  # <
    # Read case residual
    def ReadCaseResid(self):
        r"""Read a :class:`CaseResid` object

        :Call:
            >>> H = DB.ReadCaseResid()
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
        :Outputs:
            *H*: :class:`cape.cfdx.databook.CaseResid`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseResid()

    # Read case FM history
    def ReadCaseFM(self, comp):
        r"""Read a :class:`CaseFM` object

        :Call:
            >>> fm = DB.ReadCaseFM(comp)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *fm*: :class:`cape.cfdx.databook.CaseFM`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseFM(comp)

    # Read case FM history
    def ReadCaseTS(self, comp):
        r"""Read a :class:`CaseFM` object

        :Call:
            >>> fm = DB.ReadCaseFM(comp)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *fm*: :class:`cape.cfdx.databook.CaseFM`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseTS(comp)

    # Read case FM history
    def ReadCaseProp(self, comp):
        r"""Read a :class:`CaseProp` object

        :Call:
            >>> prop = DB.ReadCaseProp(comp)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *prop*: :class:`cape.cfdx.databook.CaseProp`
                Generic-property iterative history instance
        :Versions:
            * 2022-04-08 ``@ddalle``: v1.0
        """
        # Read CaseResid object from PWD
        return CaseProp(comp)
  # >

  # ========
  # Updaters
  # ========
  # <
   # -------
   # Config
   # -------
   # [
    # Process list of components
    def ProcessComps(self, comp=None, **kw):
        r"""Process list of components

        This performs several conversions:

            =============  ===================
            *comp*         Output
            =============  ===================
            ``None``       ``DB.Components``
            :class:`str`   ``comp.split(',')``
            :class:`list`  ``comp``
            =============  ===================

        :Call:
            >>> DB.ProcessComps(comp=None)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the pyCart data book class
            *comp*: {``None``} | :class:`list` | :class:`str`
                Component or list of components
        :Versions:
            * 2017-04-13 ``@ddalle``: v1.0
        """
        # Get type
        t = type(comp).__name__
        # Default list of components
        if comp is None:
            # Default: all components
            return self.Components
        elif t in ['str', 'unicode']:
            # Split by comma (also ensures list)
            return comp.split(',')
        elif t in ['list', 'ndarray']:
            # Already a list?
            return comp
        else:
            # Unknown
            raise TypeError("Cannot process component list with type '%s'" % t)
   # ]

   # ------
   # Aero
   # ------
   # [
    # Update data book
    def UpdateDataBook(self, I=None, comp=None):
        r"""Update the data book for a list of cases from the run matrix

        :Call:
            >>> DB.UpdateDataBook(I=None, comp=None)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the data book class
            *I*: :class:`list`\ [:class:`int`] | ``None``
                List of trajectory indices to update
            *comp*: {``None``} | :class:`list` | :class:`str`
                Component or list of components
        :Versions:
            * 2014-12-22 ``@ddalle``: v1.0
            * 2017-04-12 ``@ddalle``: Split by component
        """
        # Default.
        if I is None:
            # Use all trajectory points.
            I = range(self.x.nCase)
        # Process list of components
        comps = self.ProcessComps(comp)
        # Loop through components
        for comp in comps:
            # Check type
            tcomp = self.opts.get_DataBookType(comp)
            # Get handle to reader
            rdrfunc = self._readers.get(tcomp)
            # Update.
            print("%s component '%s'..." % (tcomp, comp))
            # Read the component if necessary
            if comp not in self:
                rdrfunc(self, comp, check=False, lock=False)
            # Save location
            fpwd = os.getcwd()
            os.chdir(self.RootDir)
            # Start counter
            n = 0
            # Loop through indices.
            for i in I:
                n += self[comp].UpdateCase(i, comp)
            # Return to original location
            os.chdir(fpwd)
            # Move to next component if no updates
            if n == 0:
                # Unlock
                self[comp].Unlock()
                continue
            # Status update
            print("Writing %i new or updated entries" % n)
            # Sort the component
            self[comp].Sort()
            # Write the component
            self[comp].Write(merge=True, unlock=True)

    # Function to delete entries by index
    def DeleteCases(self, I, comp=None):
        r"""Delete list of cases from data book

        :Call:
            >>> DB.Delete(I)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the pyCart data book class
            *I*: :class:`list`\ [:class:`int`]
                List of trajectory indices
            *comp*: {``None``} | :class:`list` | :class:`str`
                Component or list of components
        :Versions:
            * 2015-03-13 ``@ddalle``: v1.0
            * 2017-04-13 ``@ddalle``: Split by component
        """
        # Default.
        if I is None:
            return
        # Process list of components
        comps = self.ProcessComps(comp)
        # Loop through components
        for comp in comps:
            tcomp = self.opts.get_DataBookType(comp)
            # Get handle to reader
            rdrfunc = self._readers.get(tcomp)
            # Update.
            print("%s component '%s'..." % (tcomp, comp))
            # Read the component if necessary
            if comp not in self:
                rdrfunc(self, comp, check=False, lock=False)
            # Perform deletions
            nj = self[comp].DeleteCases(I, comp)
            # nj = self.DeleteCasesComp(I, comp)
            # Write the component
            if nj > 0:
                # Write cleaned-up data book
                self[comp].Write(unlock=True)
            else:
                # Unlock
                self[comp].Unlock()

   # ]

   # ---------
   # LineLoad
   # ---------
   # [
    # Update line load data book
    def UpdateLineLoad(self, I, comp=None, conf=None):
        r"""Update a line load data book for a list of cases

        :Call:
            >>> n = DB.UpdateLineLoad(I, comp=None, conf=None)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
            *I*: :class:`list`\ [:class:`int`]
                List of trajectory indices
            *comp*: {``None``} | :class:`str`
                Line load DataBook component or wild card
        :Outputs:
            *n*: :class:`int`
                Number of cases updated or added
        :Versions:
            * 2015-09-17 ``@ddalle``: v1.0
            * 2016-12-20 ``@ddalle``: Copied to :mod:`cape`
            * 2017-04-25 ``@ddalle``: Added wild cards
        """
        # Get list of appropriate components
        comps = self.opts.get_DataBookByGlob("LineLoad", comp)
        # Loop through those components
        for comp in comps:
            # Status update
            print("Updating LineLoad component '%s' ..." % comp)
            # Perform update and get number of deletions
            n = self.UpdateLineLoadComp(comp, I=I, conf=conf)
            # Check for updates
            if n == 0:
                # Unlock
                self.LineLoads[comp].Unlock()
                continue
            print("Added or updated %s entries" % n)
            # Write the updated results
            self.LineLoads[comp].Sort()
            self.LineLoads[comp].Write(merge=True, unlock=True)

    # Update line load data book
    def UpdateLineLoadComp(self, comp, I=None, conf=None):
        r"""Update a line load data book for a list of cases

        :Call:
            >>> n = DB.UpdateLineLoadComp(comp, conf=None, I=None)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
            *comp*: :class:`str`
                Name of line load DataBook component
            *I*: {``None``} | :class:`list`\ [:class:`int`]
                List of trajectory indices
            *qpbs*: ``True`` | {``False``}
                Whether or not to submit as a script
        :Outputs:
            *n*: :class:`int`
                Number of cases updated or added
        :Versions:
            * 2015-09-17 ``@ddalle``: v1.0
            * 2016-12-20 ``@ddalle``: Copied to :mod:`cape`
        """
        # Default case list
        if I is None:
            # Use all trajectory points
            I = range(self.x.nCase)
        # Read the line load data book if necessary
        self.ReadLineLoad(comp, conf=conf)
        # Initialize number of updates
        n = 0
        # Loop through indices.
        for i in I:
            n += self.LineLoads[comp].UpdateCase(i)
        # Ouptut
        return n

    # Function to delete entries from triqfm data book
    def DeleteLineLoad(self, I, comp=None):
        r"""Delete list of cases from LineLoad component data books

        :Call:
            >>> DB.DeleteLineLoad(I, comp=None)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the data book class
            *I*: :class:`list`\ [:class:`int`]
                List of trajectory indices
            *comp*: {``None``} | :class:`str` | :class:`list`
                Component wild card or list of component wild cards
        :Versions:
            * 2017-04-25 ``@ddalle``: v1.0
        """
        # Get list of appropriate components
        comps = self.opts.get_DataBookByGlob("LineLoad", comp)
        # Loop through those components
        for comp in comps:
            # Get number of deletions
            n = self.DeleteLineLoadComp(comp, I)
            # Check number of deletions
            if n == 0:
                continue
            # Status update
            print("%s: deleted %s LineLoad entries" % (comp, n))
            # Write the updated component
            self.LineLoads[comp].Write()

    # Function to delete line load entries
    def DeleteLineLoadComp(self, comp, I=None):
        r"""Delete list of cases from a LineLoad component data book

        :Call:
            >>> n = DB.DeleteLineLoadComp(comp, I=None)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the data book class
            *comp*: :class:`str`
                Name of component
            *I*: :class:`list`\ [:class:`int`]
                List of trajectory indices
        :Outputs:
            *n*: :class:`list`
                Number of deletions made
        :Versions:
            * 2017-04-25 ``@ddalle``: v1.0
        """
        # Default case list
        if I is None:
            # Use all trajectory points
            I = range(self.x.nCase)
        # Check type
        if self.opts.get_DataBookType(comp) != "LineLoad":
            raise ValueError(
                "Component '%s' is not a LineLoad component" % comp)
        # Read the TriqFM data book if necessary
        self.ReadLineLoad(comp)
        # Get the data book
        DBc = self.LineLoads[comp]
        # Number of cases in current data book.
        nCase = DBc.n
        # Initialize data book index array.
        J = []
        # Loop though indices to delete.
        for i in I:
            # Find the match.
            j = DBc.FindMatch(i)
            # Check if one was found.
            if j is None:
                continue
            # Append to the list of data book indices.
            J.append(j)
        # Number of deletions
        nj = len(J)
        # Exit if no deletions
        if nj == 0:
            return 0
        # Initialize mask of cases to keep.
        mask = np.ones(nCase, dtype=bool)
        # Set values equal to false for cases to be deleted.
        mask[J] = False
        # Loop through data book columns.
        for c in DBc.keys():
            # Apply the mask
            DBc[c] = DBc[c][mask]
        # Update the number of entries.
        DBc.n = len(DBc[list(DBc.keys())[0]])
        # Output
        return nj
   # ]

   # -------
   # TriqFM
   # -------
   # [
    # Update TriqFM data book
    def UpdateTriqFM(self, I, comp=None):
        r"""Update a TriqFM triangulation-extracted F&M data book

        :Call:
            >>> DB.UpdateTriqFM(I, comp=None)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
            *comp*: {``None``} | :class:`str`
                Name of TriqFM data book component (default is all)
            *I*: :class:`list`\ [:class:`int`]
                List of trajectory indices
        :Versions:
            * 2017-03-29 ``@ddalle``: v1.0
        """
        # Get list of appropriate components
        comps = self.opts.get_DataBookByGlob("TriqFM", comp)
        # Loop through those components
        for comp in comps:
            # Status update
            print("Updating TriqFM component '%s' ..." % comp)
            # Perform update and get number of deletions
            n = self.UpdateTriqFMComp(comp, I)
            # Check for updates
            if n == 0:
                # Unlock
                self.TriqFM[comp].Unlock()
                continue
            print("Added or updated %s entries" % n)
            # Write the updated results
            self.TriqFM[comp].Sort()
            self.TriqFM[comp].Write(merge=True, unlock=True)

    # Update TriqFM data book for one component
    def UpdateTriqFMComp(self, comp, I=None):
        r"""Update a TriqFM triangulation-extracted F&M data book

        :Call:
            >>> DB.UpdateTriqFMComp(comp, I=None)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
            *comp*: :class:`str`
                Name of TriqFM data book component
            *I*: {``None``} | :class:`list`\ [:class:`int`]
                List or array of run matrix indices
        :Versions:
            * 2017-03-29 ``@ddalle``: v1.0
        """
        # Default case list
        if I is None:
            # Use all trajectory points
            I = range(self.x.nCase)
        # Check type
        if self.opts.get_DataBookType(comp) != "TriqFM":
            raise ValueError(
                "Component '%s' is not a TriqFM component" % comp)
        # Read the TriqFM data book if necessary
        self.ReadTriqFM(comp, check=False, lock=False)
        # Initialize count
        n = 0
        # Loop through indices
        for i in I:
            # Update the data book for that case
            n += self.TriqFM[comp].UpdateCase(i)
        # Output
        return n

    # Function to delete entries from triqfm data book
    def DeleteTriqFM(self, I, comp=None):
        r"""Delete list of cases from TriqFM component data books

        :Call:
            >>> DB.DeleteTriqFM(I, comp=None)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the data book class
            *I*: {``None``} | :class:`list`\ [:class:`int`]
                List or array of run matrix indices
            *comp*: {``None``} | :class:`str` | :class:`list`
                Component wild card or list of component wild cards
        :Versions:
            * 2017-04-25 ``@ddalle``: v1.0
        """
        # Get list of appropriate components
        comps = self.opts.get_DataBookByGlob("TriqFM", comp)
        # Loop through those components
        for comp in comps:
            # Get number of deletions
            n = self.DeleteTriqFMComp(comp, I)
            # Check number of deletions
            if n == 0:
                # Unlock and go to next component
                self.TriqFM[comp].Unlock()
                continue
            # Status update
            print("%s: deleted %s TriqFM patch entries" % (comp, n))
            # Write the updated component
            self.TriqFM[comp].Write(unlock=True)

    # Function to delete triqfm entries
    def DeleteTriqFMComp(self, comp, I=None):
        r"""Delete list of cases from a TriqFM component data book

        :Call:
            >>> n = DB.DeleteTriqFMComp(comp, I=None)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the data book class
            *comp*: :class:`str`
                Name of component
            *I*: {``None``} | :class:`list`\ [:class:`int`]
                List or array of run matrix indices
        :Outputs:
            *n*: :class:`list`
                Number of deletions made
        :Versions:
            * 2017-04-25 ``@ddalle``: v1.0
        """
        # Default case list
        if I is None:
            # Use all trajectory points
            I = range(self.x.nCase)
        # Check type
        if self.opts.get_DataBookType(comp) != "TriqFM":
            raise ValueError(
                "Component '%s' is not a TriqFM component" % comp)
        # Read the TriqFM data book if necessary
        self.ReadTriqFM(comp, check=True, lock=True)
        # Get the data book
        DBF = self.TriqFM[comp]
        DBc = self.TriqFM[comp][None]
        # Number of cases in current data book.
        nCase = DBc.n
        # Initialize data book index array.
        J = []
        # Loop though indices to delete.
        for i in I:
            # Find the match.
            j = DBc.FindMatch(i)
            # Check if one was found.
            if j is None:
                continue
            # Append to the list of data book indices.
            J.append(j)
        # Number of deletions
        nj = len(J)
        # Exit if no deletions
        if nj == 0:
            return 0
        # Initialize mask of cases to keep.
        mask = np.ones(nCase, dtype=bool)
        # Set values equal to false for cases to be deleted.
        mask[J] = False
        # Loop through data book columns.
        for patch in DBF:
            # Get component
            DBc = DBF[patch]
            # Loop through keys
            for c in DBc.keys():
                # Apply the mask
                DBc[c] = DBc[c][mask]
            # Update the number of entries.
            DBc.n = len(DBc[list(DBc.keys())[0]])
        # Output
        return nj
   # ]

   # ----------
   # TriqPoint
   # ----------
   # [
    # Update the TriqPoint data book
    def UpdateTriqPoint(self, I, comp=None):
        r"""Update a TriqPoint triangulation-extracted point sensor data book

        :Call:
            >>> DB.UpdateTriqPoint(I, comp=None)
        :Inputs:
           *DB*: :class:`cape.cfdx.databook.DataBook`
               Instance of data book class
           *I*: :class:`list`\ [:class:`int`]
               List or array of run matrix indices
           *comp*: {``None``} | :class:`str`
               Name of TriqPoint group or all if ``None``
        :Versions:
            * 2017-10-11 ``@ddalle``: v1.0
        """
        # Get list of appropriate components
        comps = self.opts.get_DataBookByGlob("TriqPoint", comp)
        # Loop through those components
        for comp in comps:
            # Status update
            print("Updating TriqPoint group '%s' ..." % comp)
            # Perform aupdate and get number of additions
            self.UpdateTriqPointComp(comp, I)

    # Update TriqPoint data book for one component
    def UpdateTriqPointComp(self, comp, I=None):
        r"""Update a TriqPoint triangulation-extracted data book

        :Call:
            >>> n = DB.UpdateTriqPointComp(comp, I=None)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
            *comp*: {``None``} | :class:`str`
                Name of TriqPoint group or all if ``None``
            *I*: {``None``} | :class:`list`\ [:class:`int`]
                List or array of run matrix indices
        :Outputs:
            *n*: :class:`int`
                Number of updates made
        :Versions:
            * 2017-10-11 ``@ddalle``: v1.0
        """
        # Default case list
        if I is None:
            # Use all trajectory points
            I = np.arange(self.x.nCase)
        # Check type
        if self.opts.get_DataBookType(comp) != "TriqPoint":
            raise ValueError(
                "Component '%s' is not a TriqPoint component" % comp)
        # Read the TriqPoint Data book if necessary
        self.ReadTriqPoint(comp, check=False, lock=False)
        # Initialize counter
        n = 0
        # Loop through indices
        for i in I:
            # Update the data book for that case
            n += self.TriqPoint[comp].UpdateCase(i)
        # Check count
        if n > 0:
            self.TriqPoint[comp].Sort()
            print("    Added or updated %s entries" % n)
            self.TriqPoint[comp].Write(merge=True, unlock=True)
        # Output
        return n

    # Delete entries from TriqPoint data book
    def DeleteTriqPoint(self, I, comp=None):
        r"""Delete list of cases from TriqPoint component data books

        :Call:
            >>> DB.DeleteTriqPoint(I, comp=None)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the data book class
            *I*: {``None``} | :class:`list`\ [:class:`int`]
                List or array of run matrix indices
            *comp*: {``None``} | :class:`str` | :class:`list`
                Component wild card or list of component wild cards
        :Versions:
            * 2017-10-11 ``@ddalle``: v1.0
        """
        # Get list of appropriate components
        comps = self.opts.get_DataBookByGlob("TriqPoint", comp)
        # Loop through those components
        for comp in comps:
            # Delete for one component and get count
            n = self.DeleteTriqPointComp(comp, I)
            # Check number of deletions
            if n == 0:
                continue
            # Status update
            print("%s: deleted %s TriqPoint entries" % (comp, n))
            # Write the updated component (no merge)
            self.TriqPoint[comp].Write(unlock=True)

    # Delete TriqPoint individual entries
    def DeleteTriqPointComp(self, comp, I=None):
        r"""Delete list of cases from a TriqPoint component data book

        :Call:
            >>> n = DB.DeleteTriqPointComp(comp, I=None)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the data book class
            *comp*: :class:`str`
                Name of component
            *I*: {``None``} | :class:`list`\ [:class:`int`]
                List or array of run matrix indices
        :Outputs:
            *n*: :class:`list`
                Number of deletions made
        :Versions:
            * 2017-04-25 ``@ddalle``: v1.0
            * 2017-10-11 ``@ddalle``: From :func:`DeleteTriqFMComp`
        """
        # Default case list
        if I is None:
            # Use all trajectory points
            I = range(self.x.nCase)
        # Check type
        if self.opts.get_DataBookType(comp) != "TriqPoint":
            raise ValueError(
                "Component '%s' is not a TriqPoint component" % comp)
        # Read the TriqFM data book if necessary
        self.ReadTriqPoint(comp, check=True, lock=True)
        # Get the data book
        DBF = self.TriqPoint[comp]
        # Initialize total count
        n = 0
        # Loop through points
        for pt in DBF.pts:
            # Get the component
            DBc = DBF[pt]
            # Number of cases in current data book.
            nCase = len(DBc[list(DBc.keys())[0]])
            # Initialize data book index array.
            J = []
            # Loop though indices to delete.
            for i in I:
                # Find the match.
                j = DBc.FindMatch(i)
                # Check if one was found.
                if j is None:
                    continue
                # Append to the list of data book indices.
                J.append(j)
            # Number of deletions
            nj = len(J)
            # Exit if no deletions
            if nj == 0:
                continue
            # Initialize mask of cases to keep.
            mask = np.ones(nCase, dtype=bool)
            # Set values equal to false for cases to be deleted.
            mask[J] = False
            # Loop through keys
            for c in DBc.keys():
                # Apply the mask
                DBc[c] = DBc[c][mask]
            # Update the number of entries.
            DBc.n = len(DBc[list(DBc.keys())[0]])
            # Update deletion count
            n += nj
        # Output
        return n
   # ]

  # >

  # ==========
  # RunMatrix
  # ==========
  # <
    # Find an entry by run matrix variables
    def FindMatch(self, i: int) -> Optional[int]:
        r"""Find an entry by run matrix (trajectory) variables

        It is assumed that exact matches can be found.

        :Call:
            >>> j = DB.FindMatch(i)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the Cape data book class
            *i*: :class:`int`
                Index of the case from the trajectory to try match
        :Outputs:
            *j*: :class:`numpy.ndarray`\ [:class:`int`]
                Array of index(es) that match case *i* or ``NaN``
        :Versions:
            * 2016-02-27 ``@ddalle``: Added as a pointer to first component
        """
        # Get first component
        DBc = self.GetRefComponent()
        # Use its finder
        return DBc.FindMatch(i)

    # Find an entry using specified tolerance options
    def FindTargetMatch(self, DBT, i, topts, keylist='tol', **kw):
        r"""Find a target entry by run matrix (trajectory) variables

        Cases will be considered matches by comparing variables
        specified in the *topts* variable, which shares some of the
        options from the  ``"Targets"`` subsection of the ``"DataBook"``
        section of :file:`cape.json`.  Suppose that *topts* contains the
        following:

        .. code-block:: python

            {
                "RunMatrix": {"alpha": "ALPHA", "Mach": "MACH"}
                "Tolerances": {
                    "alpha": 0.05,
                    "Mach": 0.01
                },
                "Keys": ["alpha", "Mach", "beta"]
            }

        Then any entry in the data book target that matches the Mach
        number within 0.01 (using a column labeled ``"MACH"``) and alpha
        to within 0.05 is considered a match.  Because the *Keys*
        parameter contains ``"beta"``, the search will also look for
        exact matches in ``"beta"``.

        If the *Keys* parameter is not set, the search will use either
        all the keys in the trajectory, *x.cols*, or just the keys
        specified in the ``"Tolerances"`` section of *topts*.  Which of
        these two default lists to use is determined by the *keylist*
        input.

        :Call:
            >>> j = DB.FindTargetMatch(DBT, i, topts, **kw)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the Cape data book class
            *DBT*: :class:`DataBookComp` | :class:`TargetDataBook`
                Target component databook
            *i*: :class:`int`
                Index of the case from the trajectory to try match
            *topts*: :class:`dict` | :class:`TargetDataBook`
                Criteria used to determine a match
            *keylist*: ``"x"`` | {``"tol"``}
                Source for default list of keys
            *source*: {``"self"``} | ``"target"``
                Match *DB* case *i* or *DBT* case *i*
        :Outputs:
            *j*: :class:`numpy.ndarray`\ [:class:`int`]
                Array of indices that match the trajectory
        :See also:
            * :func:`cape.cfdx.databook.TargetDataBook.FindMatch`
            * :func:`cape.cfdx.databook.DataBookComp.FindMatch`
        :Versions:
            * 2016-02-27 ``@ddalle``: Added as a pointer to first component
            * 2018-02-12 ``@ddalle``: First input *x* -> *DBT*
        """
        # Get first component
        DBc = self.GetRefComponent()
        # Use its finder
        return DBc.FindTargetMatch(DBT, i, topts, keylist=keylist, **kw)

    # Match the databook copy of the trajectory
    def UpdateRunMatrix(self):
        r"""Match the trajectory to the cases in the data book

        :Call:
            >>> DB.UpdateRunMatrix()
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the Cape data book class
        :Versions:
            * 2015-05-22 ``@ddalle``: v1.0
        """
        # Get the first component.
        DBc = self.GetRefComponent()
        # Loop through the fields.
        for k in self.x.cols:
            # Copy the data.
            self.x[k] = DBc[k]
            # Set the text.
            self.x.text[k] = [str(xk) for xk in DBc[k]]
        # Set the number of cases.
        self.x.nCase = DBc.n

    # Restrict the data book object to points in the trajectory.
    def MatchRunMatrix(self):
        r"""Restrict the data book object to points in the trajectory

        :Call:
            >>> DB.MatchRunMatrix()
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the Cape data book class
        :Versions:
            * 2015-05-28 ``@ddalle``: v1.0
        """
        # Get the first component.
        DBc = self.GetRefComponent()
        # Initialize indices of points to keep.
        I = []
        J = []
        # Loop through trajectory points.
        for i in range(self.x.nCase):
            # Look for a match
            j = DBc.FindMatch(i)
            # Check for no matches.
            if j is None:
                continue
            # Match: append to both lists.
            I.append(i)
            J.append(j)
        # Loop through the trajectory keys.
        for k in self.x.cols:
            # Restrict to trajectory points that were found.
            self.x[k] = self.x[k][I]
        # Loop through the databook components.
        for comp in self.Components:
            # Loop through fields.
            for k in DBc.keys():
                # Restrict to matched cases.
                self[comp][k] = self[comp][k][J]

    # Get lists of indices of matches
    def GetTargetMatches(self, ftarg, tol=0.0, tols={}):
        r"""Get vectors of indices matching targets

        :Call:
            >>> I, J = DB.GetTargetMatches(ftarg, tol=0.0, tols={})
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
            *ftarg*: :class:`str`
                Name of the target and column
            *tol*: :class:`float`
                Tolerance for matching all keys
            *tols*: :class:`dict`
                Dictionary of specific tolerances for each key
        :Outputs:
            *I*: :class:`np.ndarray`
                Array of data book indices with matches
            *J*: :class:`np.ndarray`
                Array of target indices for each data book index
        :Versions:
            * 2015-08-30 ``@ddalle``: v1.0
        """
        # First component.
        DBC = self.GetRefComponent()
        # Initialize indices of targets *J*
        I = []
        J = []
        # Loop through cases.
        for i in np.arange(DBC.n):
            # Get the match.
            j = self.GetTargetMatch(i, ftarg, tol=tol, tols=tols)
            # Check it.
            if j is None:
                continue
            # Append it.
            I.append(i)
            J.append(j)
        # Convert to array.
        I = np.array(I)
        J = np.array(J)
        # Output
        return I, J

    # Get match for a single index
    def GetTargetMatch(self, i, ftarg, tol=0.0, tols=None):
        r"""Get index of a target match for one data book entry

        :Call:
            >>> j = DB.GetTargetMatch(i, ftarg, tol=0.0, tols={})
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
            *i*: :class:`int`
                Data book index
            *ftarg*: :class:`str`
                Name of the target and column
            *tol*: :class:`float`
                Tolerance for matching all keys
            *tols*: :class:`dict`
                Dictionary of specific tolerances for each key
        :Outputs:
            *j*: :class:`int` | ``np.nan``
                Data book target index
        :Versions:
            * 2015-08-30 ``@ddalle``: v1.0
        """
        # Default tolerances
        if tols is None:
            tols = {}
        # Check inputs
        if not isinstance(tols, dict):
            raise IOError(
                "Keyword argument *tols* to " +
                ":func:`GetTargetMatches` must be a :class:`dict`.")
        # First component.
        DBC = self.GetRefComponent()
        # Get the target.
        DBT = self.GetTargetByName(ftarg)
        # Get trajectory keys.
        tkeys = DBT.topts.get_RunMatrix()
        # Initialize constraints.
        cons = {}
        # Loop through trajectory keys
        for k in self.x.cols:
            # Get the column name.
            col = tkeys.get(k, k)
            # Continue if column not present.
            if col is None or col not in DBT:
                continue
            # Get the constraint
            cons[k] = tols.get(k, tol)
            # Set the key.
            tkeys.setdefault(k, col)
        # Initialize match indices
        m = np.arange(DBT.nCase)
        # Loop through tkeys
        for k in tkeys:
            # Get the trajectory key.
            tk = tkeys[k]
            # Make sure there's a key.
            if tk is None:
                continue
            # Check type.
            if self.x.defns[k]['Value'].startswith('float'):
                # Apply the constraint.
                m = np.intersect1d(m, np.where(
                    np.abs(DBC[k][i] - DBT[tk]) <= cons[k])[0])
            else:
                # Apply equality constraint.
                m = np.intersect1d(m, np.where(DBC[k][i] == DBT[tk])[0])
            # Check if empty; if so exit with no match.
            if len(m) == 0:
                return np.nan
        # Return the first match.
        return m[0]

    # Get match for a single index
    def GetDBMatch(self, j, ftarg, tol=0.0, tols=None):
        r"""Get index of a target match (if any) for one data book entry

        :Call:
            >>> i = DB.GetDBMatch(j, ftarg, tol=0.0, tols={})
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of a data book class
            *j*: :class:`int` | ``np.nan``
                Data book target index
            *ftarg*: :class:`str`
                Name of the target and column
            *tol*: :class:`float`
                Tolerance for matching all keys (``0.0`` enforces equality)
            *tols*: :class:`dict`
                Dictionary of specific tolerances for each key
        :Outputs:
            *i*: :class:`int`
                Data book index
        :Versions:
            * 2015-08-30 ``@ddalle``: v1.0
        """
        # Default tolerances
        if tols is None:
            tols = {}
        # Check inputs
        if not isinstance(tols, dict):
            raise IOError(
                "Keyword argument *tols* to " +
                ":func:`GetTargetMatches` must be a :class:`dict`.")
        # First component.
        DBC = self.GetRefComponent()
        # Get the target.
        DBT = self.GetTargetByName(ftarg)
        # Get trajectory keys.
        tkeys = DBT.topts.get_RunMatrix()
        # Initialize constraints.
        cons = {}
        # Loop through trajectory keys
        for k in self.x.cols:
            # Get the column name.
            col = tkeys.get(k, k)
            # Continue if column not present.
            if col is None or col not in DBT:
                continue
            # Get the constraint
            cons[k] = tols.get(k, tol)
            # Set the key.
            tkeys.setdefault(k, col)
        # Initialize match indices
        m = np.arange(DBC.n)
        # Loop through tkeys
        for k in tkeys:
            # Get the trajectory key.
            tk = tkeys[k]
            # Make sure there's a key.
            if tk is None:
                continue
            # Check type.
            if self.x.defns[k]['Value'].startswith('float'):
                # Apply the constraint.
                m = np.intersect1d(m, np.where(
                    np.abs(DBC[k] - DBT[tk][j]) <= cons[k])[0])
            else:
                # Apply equality constraint.
                m = np.intersect1d(m, np.where(DBC[k] == DBT[tk][j])[0])
            # Check if empty; if so exit with no match.
            if len(m) == 0:
                return np.nan
        # Return the first match.
        return m[0]
  # >

  # ============
  # Organization
  # ============
  # <
    # Get target to use based on target name
    def GetTargetByName(self, targ):
        r"""Get a target handle by name of the target

        :Call:
            >>> DBT = DB.GetTargetByName(targ)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the data book class
            *targ*: :class:`str`
                Name of target to find
        :Outputs:
            *DBT*: :class:`cape.cfdx.databook.TargetDataBook`
                Instance of the pyCart data book target class
        :Versions:
            * 2015-06-04 ``@ddalle``: v1.0
        """
        # Get target list
        try:
            # Get the current dict
            targs = self.Targets
        except AttributeError:
            # Not initialized
            targs = {}
        # Check for the target.
        if targ not in targs:
            # Target not found.
            raise ValueError("Target named '%s' not in data book." % targ)
        # Return the target handle.
        return targs[targ]

    # Function to sort data book
    def Sort(self, key=None, I=None):
        r"""Sort a data book according to either a key or an index

        :Call:
            >>> DB.Sort()
            >>> DB.Sort(key)
            >>> DB.Sort(I=None)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the Cape data book class
            *key*: :class:`str` | :class:`list`\ [:class:`str`]
                Name of trajectory key or list of keys on which to sort
            *I*: :class:`np.ndarray`\ [:class:`int`]
                List of indices; must have same size as data book
        :Versions:
            * 2014-12-30 ``@ddalle``: v1.0
            * 2015-06-19 ``@ddalle``: New multi-key sort
            * 2016-01-13 ``@ddalle``: Checks to allow incomplete comps
        """
        # Process inputs.
        if I is None:
            # Get first force-like component
            DBc = self.GetRefComponent()
            # Use indirect sort on the first component.
            I = DBc.ArgSort(key)
        # Loop through components.
        for comp in self.Components:
            # Check for component
            if comp not in self:
                continue
            # Check for populated component
            if self[comp].n != len(I):
                continue
            # Apply the FMDataBook.Sort() method.
            self[comp].Sort(I=I)
  # >

  # ========
  # Plotting
  # ========
  # <
    # Plot a sweep of one or more coefficients
    def PlotCoeff(self, comp, coeff, I, **kw):
        r"""Plot a sweep of one coefficients over several cases

        :Call:
            >>> h = DB.PlotCoeff(comp, coeff, I, **kw)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the data book class
            *comp*: :class:`str`
                Component whose coefficient is being plotted
            *coeff*: :class:`str`
                Coefficient being plotted
            *I*: :class:`np.ndarray`\ [:class:`int`]
                List of indexes of cases to include in sweep
        :Keyword Arguments:
            *x*: [ {None} | :class:`str` ]
                RunMatrix key for *x* axis (else plot against index)
            *Label*: {*comp*} | :class:`str`
                Manually specified label
            *Legend*: {``True``} | ``False``
                Whether or not to use a legend
            *StDev*: {``None``} | :class:`float`
                Multiple of iterative history standard deviation to plot
            *MinMax*: ``True`` | {``False``}
                Option to plot min and max from iterative history
            *Uncertainty*: ``True`` | {``False``}
                Whether to plot direct uncertainty
            *PlotOptions*: :class:`dict`
                Plot options for the primary line(s)
            *StDevOptions*: :class:`dict`
                Plot options for the standard deviation plot
            *MinMaxOptions*: :class:`dict`
                Plot options for the min/max plot
            *UncertaintyOptions*: :class:`dict`
                Dictionary of plot options for the uncertainty plot
            *FigureWidth*: :class:`float`
                Width of figure in inches
            *FigureHeight*: :class:`float`
                Height of figure in inches
            *PlotTypeStDev*: {``"FillBetween"``} | ``"ErrorBar"``
                Plot function to use for standard deviation plot
            *PlotTypeMinMax*: {``"FillBetween"``} | ``"ErrorBar"``
                Plot function to use for min/max plot
            *PlotTypeUncertainty*: ``"FillBetween"`` | {``"ErrorBar"``}
                Plot function to use for uncertainty plot
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :See also:
            * :func:`cape.cfdx.databook.DataBookComp.PlotCoeff`
        :Versions:
            * 2015-05-30 ``@ddalle``: v1.0
            * 2015-12-14 ``@ddalle``: Added error bars
        """
        # Check for the component
        if comp not in self:
            raise KeyError(
                "Data book does not contain a component '%s'" % comp)
        # Defer to the component's plot capabilities
        return self[comp].PlotCoeff(coeff, I, **kw)

    # Plot a sweep of one or more coefficients
    def PlotContour(self, comp, coeff, I, **kw):
        r"""Create a contour plot of one coefficient over several cases

        :Call:
            >>> h = DB.PlotContour(comp, coeff, I, **kw)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of the data book class
            *comp*: :class:`str`
                Component whose coefficient is being plotted
            *coeff*: :class:`str`
                Coefficient being plotted
            *I*: :class:`numpy.ndarray`\ [:class:`int`]
                List of indexes of cases to include in sweep
        :Keyword Arguments:
            *x*: :class:`str`
                RunMatrix key for *x* axis
            *y*: :class:`str`
                RunMatrix key for *y* axis
            *ContourType*: {"tricontourf"} | "tricontour" | "tripcolor"
                Contour plotting function to use
            *LineType*: {"plot"} | "triplot" | "none"
                Line plotting function to highlight data points
            *Label*: [ {*comp*} | :class:`str` ]
                Manually specified label
            *ColorBar*: [ {``True``} | ``False`` ]
                Whether or not to use a color bar
            *ContourOptions*: :class:`dict`
                Plot options to pass to contour plotting function
            *PlotOptions*: :class:`dict`
                Plot options for the line plot
            *FigureWidth*: :class:`float`
                Width of figure in inches
            *FigureHeight*: :class:`float`
                Height of figure in inches
        :Outputs:
            *h*: :class:`dict`
                Dictionary of plot handles
        :See also:
            * :func:`cape.cfdx.databook.DataBookComp.PlotCoeff`
        :Versions:
            * 2015-05-30 ``@ddalle``: v1.0
            * 2015-12-14 ``@ddalle``: Added error bars
        """
        # Check for the component
        if comp not in self:
            raise KeyError(f"Data book does not contain component '{comp}'")
        # Defer to the component's plot capabilities
        return self[comp].PlotContour(coeff, I, **kw)
  # >
    _readers = {
        "CaseProp": ReadDBCaseProp,
        "FM": ReadFM,
        "PyFunc": ReadPyFuncDataBook,
        "TimeSeries": ReadDBCompTS,
        "TriqFM": ReadTriqFM,
        "TriqPoint": _ReadTriqPoint,
        "LineLoad": ReadLineLoad,
    }


# Data book for a TriqFM component
class TriqFMDataBook(DataBook):
    r"""Force and moment component extracted from surface triangulation

    :Call:
        >>> DBF = TriqFMDataBook(x, opts, comp, RootDir=None)
    :Inputs:
        *x*: :class:`cape.runmatrix.RunMatrix`
            RunMatrix/run matrix interface
        *opts*: :class:`cape.cfdx.options.Options`
            Options interface
        *comp*: :class:`str`
            Name of TriqFM component
        *RootDir*: {``None``} | :class:`st`
            Root directory for the configuration
        *check*: ``True`` | {``False``}
            Whether or not to check LOCK status
        *lock*: ``True`` | {``False``}
            If ``True``, wait if the LOCK file exists
    :Outputs:
        *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
            Instance of TriqFM data book
    :Versions:
        * 2017-03-28 ``@ddalle``: v1.0
    """
    _triqfm_cls = TriqFMFaceDataBook
  # ======
  # Config
  # ======
  # <

    # Initialization method
    def __init__(self, x, opts, comp, **kw):
        """Initialization method

        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
        """
        # Save root directory
        self.RootDir = kw.get('RootDir', os.getcwd())
        # Save the interface
        self.x = x.Copy()
        self.opts = opts
        # Save the component
        self.comp = comp
        # Get list of patches
        self.patches = self.opts.get_DataBookPatches(comp)
        # Total list of patches including total
        self.comps = [None] + self.patches

        # Get Configuration file
        fcfg = opts.get_DataBookConfigFile(comp)
        # Default to global config file
        if fcfg is None:
            fcfg = opts.get_ConfigFile()
        # Make absolute
        if fcfg is None:
            # Ok, no file option
            self.conf = ""
        elif os.path.isabs(fcfg):
            # Already an absolute path
            self.conf = fcfg
        else:
            # Relative to root dir
            self.conf = os.path.join(self.RootDir, fcfg)
        # Restrict to triangles from *this* compID (can be list)
        self.candidateCompID = opts.get_DataBookConfigCompID(comp)
        # Loop through the patches
        for patch in self.comps:
            self[patch] = TriqFMFaceDataBook(x, opts, comp, patch=patch, **kw)

        # Reference area/length
        self.Aref = opts.get_RefArea(comp)
        self.Lref = opts.get_RefLength(comp)
        self.bref = opts.get_RefSpan(comp)
        # Moment reference point
        self.MRP = np.array(opts.get_RefPoint(comp))

    # Representation method
    def __repr__(self):
        r"""Representation method

        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
        """
        # Initialize string
        lbl = "<TriqFMDataBook %s, patches=%s>" % (self.comp, self.patches)
        # Output
        return lbl
    __str__ = __repr__

    # Read a copy
    def ReadCopy(self, check=False, lock=False):
        r"""Read a copied database object

        :Call:
            >>> DBF1 = DBF.ReadCopy(check=False, lock=False)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *check*: ``True`` | {``False``}
                Whether or not to check LOCK status
            *lock*: ``True`` | {``False``}
                If ``True``, wait if the LOCK file exists
        :Outputs:
            *DBF1*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Another instance of related TriqFM data book
        :Versions:
            * 2017-06-26 ``@ddalle``: v1.0
        """
        # Check for a name
        try:
            # Use the *name* as the first choice
            name = self.name
        except AttributeError:
            # Fall back to the *comp* attribute
            name = self.comp
        # Call the object
        DBF1 = TriqFMDataBook(self.x, self.opts, name, check=check, lock=lock)
        # Output
        return DBF1

    # Merge method
    def Merge(self, DBF1):
        """Sort point sensor group

        :Call:
            >>> DBF.Merge(DBF1)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *DBF1*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Another instance of related TriqFM data book
        :Versions:
            * 2016-06-26 ``@ddalle``: v1.0
        """
        # Check patch list
        if DBF1.patches != self.patches:
            raise KeyError("TriqFM data books have different patch lists")
        # Loop through points
        for patch in ([None] + self.patches):
            self[patch].Merge(DBF1[patch])

    # Sorting method
    def Sort(self):
        """Sort point sensor group

        :Call:
            >>> DBF.Sort()
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
        :Versions:
            * 2016-03-08 ``@ddalle``: v1.0
        """
        # Loop through points
        for patch in ([None] + self.patches):
            self[patch].Sort()

    # Output method
    def Write(self, merge=False, unlock=True):
        r"""Write to file each point sensor data book in a group

        :Call:
            >>> DBF.Write(merge=False, unlock=True)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *merge*: ``True`` | {``False``}
                Whether or not to reread data book and merge before writing
            *unlock*: {``True``} | ``False``
                Whether or not to delete any lock file
        :Versions:
            * 2015-12-04 ``@ddalle``: v1.0
            * 2017-06-26 ``@ddalle``: v1.0
        """
        # Check merge option
        if merge:
            # Read a copy
            DBF = self.ReadCopy(check=True, lock=True)
            # Merge it
            self.Merge(DBF)
            # Re-sort
            self.Sort()
        # Go to home directory
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get databook dir and triqfm dir
        fdir = self.opts.get_DataBookFolder()
        # Ensure folder exists
        if not os.path.isdir(fdir):
            os.mkdir(fdir)
        # Loop through patches
        for patch in ([None] + self.patches):
            # Sort it.
            self[patch].Sort()
            # Write it
            self[patch].Write(unlock=unlock)
        # Return to original location
        os.chdir(fpwd)

    # Lock file
    def Lock(self):
        """Lock the data book component

        :Call:
            >>> DBF.Lock()
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
        :Versions:
            * 2017-06-12 ``@ddalle``: v1.0
        """
        # Loop through patches
        for patch in ([None] + self.patches):
            # Lock each omponent
            self[patch].Lock()

    # Touch the lock file
    def TouchLock(self):
        """Touch a 'LOCK' file for a data book component to reset its mod time

        :Call:
            >>> DBF.TouchLock()
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
        :Versions:
            * 2017-06-14 ``@ddalle``: v1.0
        """
        # Loop through patches
        for patch in ([None] + self.patches):
            # Lock each omponent
            self[patch].TouchLock()

    # Lock file
    def Unlock(self):
        """Unlock the data book component (delete lock file)

        :Call:
            >>> DBF.Unlock()
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
        :Versions:
            * 2017-06-12 ``@ddalle``: v1.0
        """
        # Loop through patches
        for patch in ([None] + self.patches):
            # Lock each omponent
            self[patch].Unlock()

    # Find first force/moment component
    def GetRefComponent(self):
        r"""Get the first component

        :Call:
            >>> DBc = DBF.GetRefComponent()
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
        :Outputs:
            *DBc*: :class:`cape.cfdx.databook.FMDataBook`
                Data book for one component
        :Versions:
            * 2016-08-18 ``@ddalle``: v1.0
            * 2017-04-05 ``@ddalle``: Had to customize for TriqFM
        """
        # Get the total
        return self[None]
  # >

  # ========
  # Updaters
  # ========
  # <

    # Process a case
    def UpdateCase(self, i):
        r"""Prepare to update a TriqFM group if necessary

        :Call:
            >>> n = DBF.UpdateCase(i)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *i*: :class:`int`
                Case index
        :Outputs:
            *n*: ``0`` | ``1``
                How many updates were made
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
        """
       # -----
       # Setup
       # -----
       # (
        # Component name
        DBc = self[None]
        # Check update status
        q = True
        # Exit if no update necessary
        if not q:
            return
        # Try to find a match in the data book
        j = DBc.FindMatch(i)
        # Get the name of the folder
        frun = self.x.GetFullFolderNames(i)
        # Status update
        print(frun)
        # Go to root directory safely
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
       # )
       # ------------
       # Status Check
       # ------------
       # (
        # Check if folder exists
        if not os.path.isdir(frun):
            os.chdir(fpwd)
            return 0
        # Enter the case folder
        os.chdir(frun)
        # Determine minimum number of iterations required
        nAvg = self.opts.get_DataBookNStats(self.comp)
        nMin = self.opts.get_DataBookNMin(self.comp)
        # Get the number of iterations, etc.
        qtriq, ftriq, nStats, n0, nIter = self.GetTriqFile()
        # Process whether or not to update.
        if (not nIter) or (nIter < nMin + nAvg):
            # Not enough iterations (or zero)
            print("  Not enough iterations (%s) for analysis." % nIter)
            q = False
        elif j is None:
            # No current entry
            print("  Adding new databook entry at iteration %i." % nIter)
            q = True
        elif DBc['nIter'][j] < nIter:
            # Update
            print(
                "  Updating from iteration %i to %i." %
                (DBc['nIter'][j], nIter))
            q = True
        elif DBc['nStats'][j] < nStats:
            # Change statistics
            print("  Recomputing statistics using %i iterations." % nStats)
            q = True
        else:
            # Up-to-date
            q = False
        # Check for update
        if not q:
            os.chdir(fpwd)
            return 0
       # )
       # -----------
       # Calculation
       # -----------
       # (
        # Convert other format to TRIQ if necessary
        if qtriq:
            self.PreprocessTriq(ftriq, i=i)
        # Read the triangulation
        self.ReadTriq(ftriq)
        # Map the triangulation
        self.MapTriCompID()
        # Calculate the forces
        FM = self.GetTriqForces(i)
       # )
       # -----------------
       # Update Data Books
       # -----------------
       # (
        # Loop through patches
        for p in ([None] + self.patches):
            # Check if new case for this patch
            if j is None:
                # Increment the number of cases
                self[p].n += 1
                # Append trajectory values
                for k in self[p].xCols:
                    # Append to that column
                    self[p][k] = np.hstack((self[p][k], [self.x[k][i]]))
                # Append primary values
                for c in self[p].fCols:
                    # Get value
                    v = FM[p].get(c, np.nan)
                    # Save it.
                    self[p][c] = np.hstack((self[p][c], [v]))
                # Append iteration counts
                self[p]['nIter']  = np.hstack((self[p]['nIter'], [nIter]))
                self[p]['nStats'] = np.hstack((self[p]['nStats'], [nStats]))
            else:
                # Save updated trajectory values
                for k in self[p].xCols:
                    # Append to that column
                    self[p][k][j] = self.x[k][i]
                # Update data values
                for c in self[p].fCols:
                    # Save it.
                    self[p][c][j] = FM[p].get(c, np.nan)
                # Update the other statistics
                self[p]['nIter'][j]  = nIter
                self[p]['nStats'][j] = nStats
        # Write TRIQ/PLT/DAT file if requested
        self.WriteTriq(i, t=float(nIter))
        # Return to original folder
        os.chdir(fpwd)
        # Output
        return 1
       # )
  # >

  # ===================
  # Triq File Interface
  # ===================
  # <

    # Get file
    def GetTriqFile(self):
        r"""Get most recent ``triq`` file and its associated iterations

        :Call:
            >>> qtriq, ftriq, n, i0, i1 = DBF.GetTriqFile()
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
        :Outputs:
            *qtriq*: {``False``}
                Whether or not to convert file from other format
            *ftriq*: :class:`str`
                Name of ``triq`` file
            *n*: :class:`int`
                Number of iterations included
            *i0*: :class:`int`
                First iteration in the averaging
            *i1*: :class:`int`
                Last iteration in the averaging
        :Versions:
            * 2016-12-19 ``@ddalle``: Added to the module
        """
        # Get properties of triq file
        ftriq, n, i0, i1 = casecntl.GetTriqFile()
        # Output
        return False, ftriq, n, i0, i1

    # Convert
    def PreprocessTriq(self, ftriq, **kw):
        r"""Perform any necessary preprocessing to create ``triq`` file

        :Call:
            >>> ftriq = DBF.PreprocessTriq(ftriq, qpbs=False, f=None)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *ftriq*: :class:`str`
                Name of triq file
            *i*: {``None``} | :class:`int`
                Case index
        :Versions:
            * 2016-12-19 ``@ddalle``: v1.0
            * 2016-12-21 ``@ddalle``: Added PBS
        """
        pass

    # Read a Triq file
    def ReadTriq(self, ftriq):
        r"""Read a ``triq`` annotated surface triangulation

        :Call:
            >>> DBF.ReadTriq(ftriq)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *ftriq*: :class:`str`
                Name of ``triq`` file
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
        """
        # Delete the triangulation if present
        try:
            self.triq
            del self.triq
        except AttributeError:
            pass
        # Read using :mod:`cape`
        self.triq = trifile.Triq(ftriq, c=self.conf)
  # >

  # ============
  # Triq Writers
  # ============
  # <
    # Function to write TRIQ file if requested
    def WriteTriq(self, i, **kw):
        """Write mapped solution as TRIQ or Tecplot file with zones

        :Call:
            >>> DBF.WriteTriq(i, **kw)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *i*: :class:`int`
                Case index
            *t*: {``1``} | :class:`float`
                Iteration number
        :Versions:
            * 2017-03-30 ``@ddalle``: v1.0
        """
        # Get the output file type
        fmt = self.opts.get_DataBookOutputFormat(self.comp)
        # List of known formats
        fmts = ["tri", "triq", "plt", "dat"]
        # Check the option
        if fmt is None:
            # Nothing more to do
            return
        elif type(fmt).__name__ not in ["unicode", "str"]:
            # Bad type
            raise TypeError(
                ('Invalid "OutputFormat": %s (type %s)' % (fmt, type(fmt))) +
                (' for TriqFM component "%s"' % self.comp))
        elif fmt.lower() not in fmts:
            # Not known
            print("    Unknown TRIQ output format '%s'" % fmt)
            print('    Available options are "triq", "plt", and "data"')
            return
        # Go to data book folder safely
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        os.chdir(self.opts.get_DataBookFolder())
        # Enter the "triqfm" folder (create if needed)
        if not os.path.isdir("triqfm"):
            os.mkdir("triqfm")
        os.chdir("triqfm")
        # Get the group and run folders
        fgrp = self.x.GetGroupFolderNames(i)
        frun = self.x.GetFullFolderNames(i)
        # Create folders if needed
        if not os.path.isdir(fgrp):
            os.mkdir(fgrp)
        if not os.path.isdir(frun):
            os.mkdir(frun)
        # Go into the run folder
        os.chdir(frun)
        # Name of file
        fpre = self.opts.get_DataBookPrefix(self.comp)
        # Convert the file as needed
        if fmt.lower() in ["tri", "triq"]:
            # Down select the mapped patches
            triq = self.SelectTriq()
            # Write the TRIQ in this format
            triq.Write("%s.triq" % fpre, ascii=True)
        elif fmt.lower() == "dat":
            # Create Tecplot PLT interface
            pltq = self.Triq2Plt(self.triq, i=i, **kw)
            # Write ASCII file
            pltq.WriteDat("%s.dat" % fpre)
            # Delete it
            del pltq
        elif fmt.lower() == "plt":
            # Create Tecplot PLT interface
            pltq = self.Triq2Plt(self.triq, i=i, **kw)
            # Write binary file
            pltq.Write("%s.plt" % fpre)
            # Delete it
            del pltq
        # Go back to original location
        os.chdir(fpwd)

    # Get the component numbers of the mapped patches
    def GetPatchCompIDs(self):
        r"""Get the list of component IDs mapped from the template *tri*

        :Call:
            >>> CompIDs = DBF.GetPatchCompIDs()
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
        :Outputs:
            *CompIDs*: :class:`list`\ [:class:`int`] | ``None``
                List of component IDs that came from the mapping file
        :Versions:
            * 2017-03-30 ``@ddalle``: v1.0
        """
        # Initialize list of Component IDs
        CompIDs = []
        # Loop through the patches
        for patch in self.patches:
            # Get the component for this patch
            compID = self.GetCompID(patch)
            # Check the type
            t = type(compID).__name__
            # Check if it's a string
            if t in ['str', 'unicode']:
                # Get the component ID from the *triq*
                try:
                    # Get the value from *triq.config* or *triq.Conf*
                    comp = self.triq.GetCompID(compID)
                    # Check if it's a list
                    if type(comp).__name__ in ["list", "ndarray"]:
                        # Check for list
                        if len(comp) > 1:
                            raise ValueError(
                                ("Component ID %s for patch '%s'"
                                    % (comp, patch)) +
                                (" is not a integer or singleton"))
                        # Get the one element
                        CompIDs.append(comp[0])
                    else:
                        # Append the integer
                        CompIDs.append(comp)
                except Exception:
                    # Unknown component
                    raise ValueError(
                        "Could not determine component ID for patch '%s'"
                        % patch)
            else:
                # If it was specified numerically, check the *compmap*
                # If the mapping had to renumber the component, it will be
                # in this dictionary; otherwise use the compID as is.
                CompIDs.append(self.compmap.get(compID, compID))
        # Output
        self.CompIDs = CompIDs
        return CompIDs

    # Select the relevant components of the mapped TRIQ file
    def SelectTriq(self):
        """Select the components of *triq* that are mapped patches

        :Call:
            >>> triq = DBF.SelectTriq()
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
        :Outputs:
            *triq*: :class:`cape.trifile.Triq`
                Interface to annotated surface triangulation
        :Versions:
            * 2017-03-30 ``@ddalle``: v1.0
        """
        # Get component IDs
        CompIDs = self.GetPatchCompIDs()
        # Downselect
        triq = self.triq.GetSubTri(CompIDs)
        # Output
        return triq

    # Convert the TRIQ file
    def Triq2Plt(self, triq, **kw):
        r"""Convert an annotated tri (TRIQ) interface to Tecplot (PLT)

        :Call:
            >>> plt = DBF.Triq2Plt(triq, **kw)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *triq*: :class:`cape.trifile.Triq`
                Interface to annotated surface triangulation
            *i*: {``None``} | :class:`int`
                Index number if needed
            *t*: {``1.0``} | :class:`float`
                Time step or iteration number
        :Outputs:
            *plt*: :class:`cape.plt.Plt`
                Binary Tecplot interface
        :Versions:
            * 2017-03-30 ``@ddalle``: v1.0
        """
        # Get component IDs
        CompIDs = self.GetPatchCompIDs()
        # Get freestream conditions
        if 'i' in kw:
            # Get freestream conditions
            kwfm = self.GetConditions(kw["i"])
            # Set those conditions
            for k in kwfm:
                kw.setdefault(k, kwfm[k])
        # Perform conversion
        pltq = pltfile.Plt(triq=triq, CompIDs=CompIDs, **kw)
        # Output
        return pltq
  # >

  # ========
  # Mapping
  # ========
  # <

    # Get compID option for a patch
    def GetCompID(self, patch):
        r"""Get the component ID name(s) or number(s) to use for each patch

        :Call:
            >>> compID = DBF.GetCompID(patch)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *patch*: :class:`str`
                Name of patch
        :Outputs:
            *compID*: {*patch*} | :class:`str` | :class:`int` | :class:`list`
                Name, number, or list thereof of *patch* in map tri file
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
        """
        # Get data book option
        compIDmap = self.opts.get_DataBookCompID(self.comp)
        # Get the type
        t = type(compIDmap).__name__
        # Behavior based on type
        if compIDmap is None:
            # Use the patch name
            return patch
        elif t in ['dict']:
            # Custom dictionary (default to *patch*)
            return compIDmap.get(patch, patch)
        elif (t in ['list']):
            # List of comps for the whole TRI
            return compIDmap
        else:
            # Give up
            return patch

    # Read the map file
    def ReadTriMap(self):
        r"""Read the triangulation to use for mapping

        :Call:
            >>> DBF.ReadTriMap()
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
        """
        # Get the name of the tri file and configuration
        ftri = self.opts.get_DataBookMapTri(self.comp)
        fcfg = self.opts.get_DataBookConfigFile(self.comp)
        # Check for absolute paths
        if (ftri) and (not os.path.isabs(ftri)):
            # Read relative to *RootDir*
            ftri = os.path.join(self.RootDir, ftri)
        # Repeat for configuration
        if (fcfg) and (not os.path.isabs(fcfg)):
            # Read relative to *RootDir*
            fcfg = os.path.join(self.RootDir, fcfg)
        # Save triangulation value
        if ftri:
            # Read the triangulation
            self.tri = trifile.Tri(ftri, c=fcfg)
        else:
            # No triangulation map
            self.tri = None

    # Map the components
    def MapTriCompID(self):
        r"""Perform any component ID mapping if necessary

        :Call:
            >>> DBF.MapTriCompID()
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
        :Attributes:
            *DBF.compmap*: :class:`dict`
                Map of component numbers altered during the mapping
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
        """
        # Ensure tri is present
        try:
            self.tri
        except Exception:
            self.ReadTriMap()
        # Check for a tri file
        if self.tri is None:
            self.compmap = {}
        else:
            ftri = self.opts.get_DataBookMapTri(self.comp)
            print("    Mapping component IDs using '%s'" % ftri)
            # Get tolerances
            kw = {"AbsTol": self.opts.get_DataBookAbsTol(self.comp)}
            # Set candidate component ID
            kw["compID"] = self.candidateCompID
            try:
                # Eliminate unused component names, if any
                self.triq.RestrictConfigCompID()
            except Exception:
                pass
            # Map the component IDs
            self.compmap = self.triq.MapTriCompID(self.tri, **kw)
  # >

  # ===========================
  # Force & Moment Computation
  # ===========================
  # <
    # Get relevant freestream conditions
    def GetConditions(self, i):
        r"""Get the freestream conditions needed for forces

        :Call:
            >>> xi = DBF.GetConditions(i)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *i*: :class:`int`
                Case index
        :Outputs:
            *xi*: :class:`dict`
                Dictionary of Mach number (*mach*), Reynolds number (*Re*)
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
        """
        # Attempt to get Mach number
        try:
            # Use the trajectory
            mach = self.x.GetMach(i)
        except Exception:
            # No Mach number specified in run matrix
            raise ValueError(
                ("Could not determine freestream Mach number\n") +
                ("TriqFM component '%s'" % self.comp))
        # Attempt to get Reynolds number (not needed if inviscid)
        try:
            # Use the trajectory
            Rey = self.x.GetReynoldsNumber(i)
        except Exception:
            # Assume it's not needed
            Rey = 1.0
        # Ratio of specific heats
        gam = self.x.GetGamma(i)
        # Dynamic pressure
        q = self.x.GetDynamicPressure(i)
        # Output
        return {"mach": mach, "Re": Rey, "gam": gam, "q": q}

    # Calculate forces and moments
    def GetTriqForcesPatch(self, patch, i, **kw):
        r"""Get the forces and moments on a patch

        :Call:
            >>> fm = DBF.GetTriqForces(patch, i, **kw)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *patch*: :class:`str`
                Name of patch
            *i*: :class:`int`
                Case index
        :Outputs:
            *fm*: :class:`dict`\ [:class:`float`]
                Dictionary of force & moment coefficients
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
        """
        # Set inputs for TriqForces
        kwfm = self.GetConditions(i)
        # Apply remaining options
        kwfm["Aref"] = self.Aref
        kwfm["Lref"] = self.Lref
        kwfm["bref"] = self.bref
        kwfm["MRP"]  = self.MRP
        kwfm["incm"] = self.opts.get_DataBookMomentum(self.comp)
        kwfm["gauge"] = self.opts.get_DataBookGauge(self.comp)
        # Get component for this patch
        compID = self.GetCompID(patch)
        # Default list: the whole protuberance
        if compID is None:
            # Get list from TRI
            compID = np.unique(self.tri.CompID)
        # Perform substitutions if necessary
        if type(compID).__name__ in ['list', 'ndarray']:
            # Loop through components
            for i in range(len(compID)):
                # Get comp
                compi = compID[i]
                # Check for int
                if type(compi).__name__ != "int":
                    continue
                # Check the component number mapping
                compID[i] = self.compmap.get(compi, compi)
        # Calculate forces
        FM = self.triq.GetTriForces(compID, **kwfm)
        # Apply transformations
        FM = self.ApplyTransformations(i, FM)
        # Get dimensional forces if requested
        FM = self.GetDimensionalForces(patch, i, FM)
        # Get additional states
        FM = self.GetStateVars(patch, FM)
        # Output
        return FM

    # Get other forces
    def GetDimensionalForces(self, patch, i, FM):
        r"""Get dimensional forces

        This dimensionalizes any force or moment coefficient already in *fm*
        replacing the first character ``'C'`` with ``'F'``.  For example,
        ``"FA"`` is the dimensional axial force from ``"CA"``, and ``"FAv"`` is
        the dimensional axial component of the viscous force

        :Call:
            >>> fm = DBF.GetDimensionalForces(patch, i, FM)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *patch*: :class:`str`
                Name of patch
            *i*: :class:`int`
                Case index
            *fm*: :class:`dict`\ [:class:`float`]
                Dictionary of force & moment coefficients
        :Outputs:
            *fm*: :class:`dict`\ [:class:`float`]
                Dictionary of force & moment coefficients
        :Versions:
            * 2017-03-29 ``@ddalle``: v1.0
        """
        # Dimensionalization value
        Fref = self.x.GetDynamicPressure(i) * self.Aref
        # Loop through float columns in the data book
        for k in self[patch].fCols:
            # Skip if already present in *fm*
            if k in FM:
                continue
            # Check if it's a dimensional force
            if not k.startswith('F'):
                continue
            # Get the force name
            f = k[1:]
            # Assemble the apparent coefficient name
            c = 'C' + f
            # Check if it's present in non-dimensional form
            if c not in FM:
                continue
            # Filter the component to see if it's a moment
            if f.startswith('LL'):
                # Use reference span for rolling moment
                FM[k] = FM[c]*Fref*self.bref
            elif f.startswith('LN'):
                # Use reference span for yawing moment
                FM[k] = FM[c]*Fref*self.bref
            elif f.startswith('LM'):
                # Use reference chord for pitching moment
                FM[k] = FM[c]*Fref*self.Lref
            else:
                # Assume force
                FM[k] = FM[c]*Fref
        # Output for clarity
        return FM

    # Get other stats
    def GetStateVars(self, patch, FM):
        r"""Get additional state variables, such as minimum *Cp*

        :Call:
            >>> fm = DBF.GetStateVars(patch, FM)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *patch*: :class:`str`
                Name of patch
            *fm*: :class:`dict`\ [:class:`float`]
                Dictionary of force & moment coefficients
        :Outputs:
            *fm*: :class:`dict`\ [:class:`float`]
                Dictionary of force & moment coefficients
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
        """
        # Get component for this patch
        compID = self.GetCompID(patch)
        # Get nodes for that compID(s)
        I = self.triq.GetNodesFromCompID(compID)
        if len(I) == 0:
            raise ValueError(
                "Patch '%s' (compID=%s) has no triangles" %
                (patch, compID))
        # Loop through float columns
        for c in self[patch].fCols:
            # Skip if already in *fm*
            if c in FM:
                continue
            # Check if it's something we recognize
            if c.lower() in ['cpmin', 'cp_min']:
                # Get minimum value from first column
                FM[c] = np.min(self.triq.q[I, 0])
            elif c.lower() in ['cpmax', 'cp_max']:
                # Get maximum value from first column
                FM[c] = np.max(self.triq.q[I, 0])
            elif c.lower() in ['cp', 'cp_mu', 'cp_mean']:
                # Mean *Cp*
                FM[c] = np.mean(self.triq.q[I, 0])
        # Output for clarity
        return FM

    # Get all patches
    def GetTriqForces(self, i, **kw):
        r"""Get the forces, moments, and other states on each patch

        :Call:
            >>> fm = DBF.GetTriqForces(i)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *i*: :class:`int`
                Case index
        :Outputs:
            *fm*: :class:`dict` (:class:`dict`\ [:class:`float`])
                Dictionary of force & moment dictionaries for each patch
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
        """
        # Initialize dictionary of forces
        FM = {}
        # List of patches
        if self.patches:
            # Use the patch list as per usual
            patches = self.patches
        else:
            # Use the component
            patches = [None]
        # Loop through patches
        for patch in patches:
            # Calculate forces
            FM[patch] = self.GetTriqForcesPatch(patch, i, **kw)
        # Exit if no patches
        if None in FM:
            return FM
        # Initialize cumulative sum
        FM0 = dict(FM[patch])
        # Accumulate each patch
        for patch in patches[:-1]:
            # Loop through keys
            for k in FM[patch]:
                # Accumulate the value
                if k.endswith('_min'):
                    # Take overall min
                    FM0[k] = min(FM0[k], FM[patch][k])
                elif k.endswith('_max'):
                    # Take overall max
                    FM0[k] = max(FM0[k], FM[patch][k])
                else:
                    # Add up values
                    FM0[k] += FM[patch][k]
        # Save the value; ``None`` cannot conflict because it's not a string
        FM[None] = FM0
        # Output
        return FM

    # Apply all transformations
    def ApplyTransformations(self, i, FM):
        r"""Apply transformations to forces and moments

        :Call:
            >>> fm = DBF.ApplyTransformations(i, FM)
        :Inputs:
            *DBF*: :class:`cape.cfdx.databook.TriqFMDataBook`
                Instance of TriqFM data book
            *i*: :class:`int`
                Case index
            *fm*: :class:`dict` (:class:`dict`\ [:class:`float`])
                Dictionary of force & moment coefficients
        :Outputs:
            *fm*: :class:`dict` (:class:`dict`\ [:class:`float`])
                Dictionary of transformed force & moment coefficients
        :Versions:
            * 2017-03-29 ``@ddalle``: v1.0
        """
        # Get the data book transformations for this component
        db_transforms = self.opts.get_DataBookTransformations(self.comp)
        # Special transformation to reverse *CLL* and *CLN*
        tflight = {"Type": "ScaleCoeffs", "CLL": -1.0, "CLN": -1.0}
        # Check for ScaleCoeffs
        if tflight not in db_transforms:
            # Append a transformation to reverse *CLL* and *CLN*
            db_transforms.append(tflight)
        # Loop through the transformations
        for topts in db_transforms:
            # Apply transformation type
            FM = self.TransformFM(FM, topts, i)
        # Output for clarity
        return FM

    # Transform force or moment reference frame
    def TransformFM(self, FM, topts, i):
        r"""Transform a force and moment history

        Available transformations and their parameters are listed below.

            * "Euler321": "psi", "theta", "phi"
            * "ScaleCoeffs": "CA", "CY", "CN", "CLL", "CLM", "CLN"

        RunMatrix variables are used to specify values to use for the
        transformation variables.  For example,

            .. code-block:: python

                topts = {"Type": "Euler321",
                    "psi": "Psi", "theta": "Theta", "phi": "Phi"}

        will cause this function to perform a reverse Euler 3-2-1
        transformation using *x.Psi[i]*, *x.Theta[i]*, and *x.Phi[i]*
        as the angles.

        Coefficient scaling can be used to fix incorrect reference areas
        or flip axes. The default is actually to flip *CLL* and *CLN*
        due to the transformation from CFD axes to standard flight
        dynamics axes.

            .. code-block:: python

                topts = {"Type": "ScaleCoeffs",
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
                The index of the case to in the current run matrix
        :Versions:
            * 2014-12-22 ``@ddalle``: v1.0
        """
        # Get the transformation type.
        ttype = topts.get("Type", "")
        # Check it.
        if ttype in ["Euler321", "Euler123"]:
            # Get the angle variable names.
            # Use same as default in case it's obvious what they should be.
            kph = topts.get('phi', 0.0)
            kth = topts.get('theta', 0.0)
            kps = topts.get('psi', 0.0)
            # Extract roll
            if type(kph).__name__ not in ['str', 'unicode']:
                # Fixed value
                phi = kph*deg
            elif kph.startswith('-'):
                # Negative roll angle.
                phi = -self.x[kph[1:]][i]*deg
            else:
                # Positive roll
                phi = self.x[kph][i]*deg
            # Extract pitch
            if type(kth).__name__ not in ['str', 'unicode']:
                # Fixed value
                theta = kth*deg
            elif kth.startswith('-'):
                # Negative pitch
                theta = -self.x[kth[1:]][i]*deg
            else:
                # Positive pitch
                theta = self.x[kth][i]*deg
            # Extract yaw
            if type(kps).__name__ not in ['str', 'unicode']:
                # Fixed value
                psi = kps*deg
            elif kps.startswith('-'):
                # Negative yaw
                psi = -self.x[kps[1:]][i]*deg
            else:
                # Positive pitch
                psi = self.x[kps][i]*deg
            # Sines and cosines
            cph = np.cos(phi)
            cth = np.cos(theta)
            cps = np.cos(psi)
            sph = np.sin(phi)
            sth = np.sin(theta)
            sps = np.sin(psi)
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
            # Area transformations
            if "Ay" in FM:
                # Assemble area vector
                Ac = np.array([FM["Ax"], FM["Ay"], FM["Az"]])
                # Transform
                Ab = np.dot(R, Ac)
                # Reset
                FM["Ax"] = Ab[0]
                FM["Ay"] = Ab[1]
                FM["Az"] = Ab[2]
            # Force transformations
            # Loop through suffixes
            for s in ["", "p", "vac", "v", "m"]:
                # Construct force coefficient names
                cx = "CA" + s
                cy = "CY" + s
                cz = "CN" + s
                # Check if the coefficient is present
                if cy in FM:
                    # Assemble forces
                    Fc = np.array([FM[cx], FM[cy], FM[cz]])
                    # Transform
                    Fb = np.dot(R, Fc)
                    # Reset
                    FM[cx] = Fb[0]
                    FM[cy] = Fb[1]
                    FM[cz] = Fb[2]
                # Construct moment coefficient names
                cx = "CLL" + s
                cy = "CLM" + s
                cz = "CLN" + s
                # Check if the coefficient is present
                if cy in FM:
                    # Assemble moment vector
                    Mc = np.array([FM[cx], FM[cy], FM[cz]])
                    # Transform
                    Mb = np.dot(R, Mc)
                    # Reset
                    FM[cx] = Mb[0]
                    FM[cy] = Mb[1]
                    FM[cz] = Mb[2]

        elif ttype in ["ScaleCoeffs"]:
            # Loop through coefficients.
            for c in topts:
                # Get the value.
                k = topts[c]
                # Check if it's a number.
                if type(k).__name__ not in ["float", "int"]:
                    # Assume they meant to flip it.
                    k = -1.0
                # Loop through suffixes
                for s in ["", "p", "vac", "v", "m"]:
                    # Construct overall name
                    cc = c + s
                    # Check if it's present
                    if cc in FM:
                        FM[cc] = k*FM[cc]
        else:
            raise IOError(
                "Transformation type '%s' is not recognized." % ttype)
        # Output for clarity
        return FM
  # >


def _cummin_r(arr: np.ndarray) -> np.ndarray:
    r"""Calculate reversed cumulative minimum of a 1D array

    :Call:
        >>> v = _cummin_r(arr)
    """
    # Calculate cumulative minimum in revers
    return np.flip(np.minimum.accumulate(np.flip(arr)))


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
