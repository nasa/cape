r"""
``cape.pyover.databook``: DataBook module for OVERFLOW
=======================================================

This module contains functions for reading and processing forces,
moments, and other statistics from cases in a trajectory. Data books are
usually created by using the :func:`cape.pyover.cntl.Cntl.ReadDataBook`
function.

    .. code-block:: python

        # Read OVERFLOW control instance
        cntl = pyOver.Cntl("pyOver.json")
        # Read the data book
        cntl.ReadDataBook()
        # Get a handle
        DB = cntl.DataBook

        # Read a line load component
        DB.ReadLineLoad("CORE_LL")
        DBL = DB.LineLoads["CORE_LL"]
        # Read a target
        DB.ReadTarget("t97")
        DBT = DB.Targets["t97"]

Data books can be created without an overall control structure, but it
requires creating a run matrix object using
:class:`cape.pyover.runmatrix.RunMatrix`, so it is a more involved
process.

Data book modules are also invoked during update and reporting
command-line calls.

    .. code-block:: console

        $ pyfun --aero
        $ pyfun --ll
        $ pyfun --triqfm
        $ pyfun --report

The available components mirror those described on the template data
book modules, :mod:`cape.cfdx.databook`, :mod:`cape.cfdx.lineload`, and
:mod:`cape.cfdx.pointsensor`.  However, some data book types may not be
implemented for all CFD solvers.

:See Also:
    * :mod:`cape.cfdx.databook`
    * :mod:`cape.cfdx.lineload`
    * :mod:`cape.cfdx.pointsensor`
    * :mod:`cape.pyover.lineload`
    * :mod:`cape.options.DataBook`
    * :mod:`cape.pyover.options.DataBook`
"""

# Standard library
import os
from io import IOBase

# Third-party modules
import numpy as np

# Local imports
from . import casecntl
from . import pointsensor
from . import lineload
from ..cfdx import databook
from ..dkit import basedata


# Read component names from a fomoco file
def read_fomoco_comps(fname: str):
    r"""Get list of components in an OVERFLOW fomoco file

    :Call:
        >>> comps = read_fomoco_comps(fp)
    :Inputs:
        *fname*: :class:`str`
            Name of the file to read
    :Outputs:
        *comps*: :class:`list`\ [:class:`str`]
            List of components
    :Versions:
        * 2016-02-03 ``@ddalle``: v1.0 (ReadFomocoCOmps)
        * 2024-01-24 ``@ddalle``: v1.1; handle instead of file name
    """
    # Initialize components
    comps = []
    # Open file
    with open(fname, 'rb') as fp:
        # Read the first line and first component.
        comp = fp.readline().strip().decode("ascii")
        # Loop until a component repeats
        while comp not in comps:
            # Check for empty line
            if comp == "":
                break
            # Add the component
            comps.append(comp)
            # Move to the next component
            fp.seek(569, 1)
            # Read the next component.
            comp = fp.readline().strip().decode("ascii")
    # Output
    return comps


# Read basic stats from a fomoco file
def read_fomoco_niter(fname, ncomp: int):
    r"""Get number of iterations in an OVERFLOW fomoco file

    :Call:
        >>> nIter = read_fomoco_niter(fname)
        >>> nIter = read_fomoco_niter(fname, nComp)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *nComp*: :class:`int` | ``None``
            Number of components in each record
    :Outputs:
        *nIter*: :class:`int`
            Number of iterations in the file
    :Versions:
        * 2016-02-03 ``@ddalle``: v1.0
        * 2024-01-24 ``@ddalle``: v1.1; require *ncomp*
    """
    # Open file to get number of iterations
    with open(fname, 'r') as fp:
        # Go to end of file to get length of file
        fp.seek(0, 2)
        # Position at EOF
        filesize = fp.tell()
    # Save number of iterations
    return int(np.ceil(filesize / (ncomp*650.0)))


# Read grid names from a resid file
def ReadResidGrids(fname):
    r"""Get list of grids in an OVERFLOW residual file

    :Call:
        >>> grids = ReadResidGrids(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *grids*: :class:`list`\ [:class:`str`]
            List of grids
    :Versions:
        * 2016-02-04 ``@ddalle``: v1.0
    """
    # Initialize grids
    comps = []
    # Open the file
    f = open(fname, 'r')
    # Read the first line
    line = f.readline()
    # Get component name
    try:
        comp = ' '.join(line.split()[14:])
    except Exception:
        comp = line.split()[-1]
    # Loop until a component repeates
    while len(comps) == 0 or comp != comps[0]:
        # Add the component
        comps.append(comp)
        # Read the next line
        line = f.readline()
        # Get component name
        try:
            comp = ' '.join(line.split()[14:])
        except Exception:
            comp = line.split()[-1]
    # Close the file
    f.close()
    # Output
    return comps


# Read number of grids from a resid file
def ReadResidNGrids(fname):
    r"""Get number of grids from an OVERFLOW residual file

    :Call:
        >>> nGrid = ReadResidNGrids(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *nGrid*: :class:`int`
            Number of grids
    :Versions:
        * 2016-02-04 ``@ddalle``: v1.0
        * 2024-01-24 ``@ddalle``: v1.1; context manager
    """
    # Initialize number of grids
    nGrid = 1
    # Open the file
    with open(fname, 'r') as fp:
        # Discard first line (always grid 1)
        fp.readline()
        # Loop until grid number decreases
        while True:
            # Read next line
            line = fp.readline()
            # Check for EOF
            if line == '':
                break
            # Read grid number
            iGrid = int(line.split(maxsplit=1)[0])
            # If it's grid 1; we reached the end
            if iGrid == 1:
                break
            # Update grid count
            nGrid += 1
    # Output
    return nGrid


# Read the first iteration number from a resid file.
def ReadResidFirstIter(fname):
    r"""Read the first iteration number in an OVERFLOW residual file

    :Call:
        >>> iIter = ReadResidFirstIter(fname)
        >>> iIter = ReadResidFirstIter(f)
    :Inputs:
        *fname*: :class:`str`
            Name of file to query
        *f*: :class:`file`
            Already opened file handle to query
    :Outputs:
        *iIter*: :class:`int`
            Iteration number from first line
    :Versions:
        * 2016-02-04 ``@ddalle``: v1.0
        * 2023-01-14 ``@ddalle``: v1.1; fix file type check for py3
    """
    # Check input type
    if isinstance(fname, IOBase):
        # Already a file
        fp = fname
        # Check if it's open already
        qf = True
        # Get current location
        ft = fp.tell()
    else:
        # Open the file.
        fp = open(fname, 'r')
        # Not open
        qf = False
    # Read the second entry from the first line
    iIter = int(fp.readline().split()[1])
    # Close the file.
    if qf:
        # Return to original location
        fp.seek(ft)
    else:
        # Close the file
        fp.close()
    # Output
    return iIter


# Get number of iterations from a resid file
def ReadResidNIter(fname):
    r"""Get number of iterations in an OVERFLOW residual file

    :Call:
        >>> nIter = ReadResidNIter(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to query
    :Outputs:
        *nIter*: :class:`int`
            Number of iterations
    :Versions:
        * 2016-02-04 ``@ddalle``: v1.0
        * 2022-01-09 ``@ddalle``: v1.1; Python 3 int division
    """
    # Get the number of grids.
    nGrid = ReadResidNGrids(fname)
    # Open the file
    f = open(fname, 'r')
    # Go to the end of the file
    f.seek(0, 2)
    # Use the position to determine the number of lines
    nIter = f.tell() // nGrid // 218
    # close the file.
    f.close()
    # Output
    return nIter


# Component data book
class FMDataBook(databook.FMDataBook):
    r"""Individual component data book

    This class is derived from :class:`cape.cfdx.databook.DataBookComp`.

    :Call:
        >>> DBc = DBComp(comp, x, opts)
    :Inputs:
        *comp*: :class:`str`
            Name of the component
        *x*: :class:`pyOver.runmatrix.RunMatrix`
            RunMatrix for processing variable types
        *opts*: :class:`pyOver.options.Options`
            Global pyCart options instance
        *targ*: {``None``} | :class:`str`
            If used, read a duplicate data book as a target named *targ*
    :Outputs:
        *DBc*: :class:`DBComp`
            An individual component data book
    :Versions:
        * 2016-09-15 ``@ddalle``: v1.0
    """
    # Read case FM history
    def ReadCase(self, comp):
        r"""Read a :class:`CaseFM` object

        :Call:
            >>> fm = DB.ReadCase(comp)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *fm*: :class:`CaseFM`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: v1.0
            * 2023-07-10 ``@ddalle``: v1.1; use ``CaseRunner``
        """
        # Get a case runner
        runner = casecntl.CaseRunner()
        # Get the phase number
        k = runner.get_phase()
        # Appropriate prefix
        proj = self.opts.get_Prefix(k)
        # Read CaseResid object from PWD
        return CaseFM(proj, comp)

    # Read case residual
    def ReadCaseResid(self):
        r"""Read a :class:`CaseResid` object

        :Call:
            >>> hist = DB.ReadCaseResid()
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
        :Outputs:
            *hist*: :class:`CaseResid`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: v1.0
            * 2023-07-10 ``@ddalle``: v1.1; use ``CaseRunner``
        """
        # Get a case runner
        runner = casecntl.CaseRunner()
        # Get the phase number
        k = runner.get_phase()
        # Appropriate prefix
        proj = self.opts.get_Prefix(k)
        # Read CaseResid object from PWD
        return CaseResid(proj)


class PropDataBook(databook.PropDataBook):
    # Read case residual
    def ReadCaseResid(self):
        r"""Read a :class:`CaseResid` object

        :Call:
            >>> hist = DB.ReadCaseResid()
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
        :Outputs:
            *hist*: :class:`CaseResid`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: v1.0
            * 2023-07-10 ``@ddalle``: v1.1; use ``CaseRunner``
        """
        # Get a case runner
        runner = casecntl.CaseRunner()
        # Get the phase number
        k = runner.get_phase()
        # Appropriate prefix
        proj = self.opts.get_Prefix(k)
        # Read CaseResid object from PWD
        return CaseResid(proj)


class PyFuncDataBook(databook.PyFuncDataBook):
    pass


# Data book target instance
class TargetDataBook(databook.TargetDataBook):
    pass


class TimeSeriesDataBook(databook.TimeSeriesDataBook):
    # Read case residual
    def ReadCaseResid(self):
        r"""Read a :class:`CaseResid` object

        :Call:
            >>> hist = DB.ReadCaseResid()
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
        :Outputs:
            *hist*: :class:`CaseResid`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: v1.0
            * 2023-07-10 ``@ddalle``: v1.1; use ``CaseRunner``
        """
        # Get a case runner
        runner = casecntl.CaseRunner()
        # Get the phase number
        k = runner.get_phase()
        # Appropriate prefix
        proj = self.opts.get_Prefix(k)
        # Read CaseResid object from PWD
        return CaseResid(proj)


# Force/moment history
class CaseFM(databook.CaseFM):
    r"""Force and moment iterative histories

    This class contains methods for reading data about an the history of
    an individual component for a single casecntl. It reads the Tecplot
    file ``$proj_fm_$comp.dat`` where *proj* is the lower-case root
    project name and *comp* is the name of the component. From this file
    it determines which coefficients are recorded automatically.

    :Call:
        >>> fm = CaseFM(proj, comp)
    :Inputs:
        *proj*: :class:`str`
            Root name of the project
        *comp*: :class:`str`
            Name of component to process
    :Outputs:
        *fm*: :class:`CaseFM`
            Force and moment iterative history instance
    """
    # Attributes
    __slots__ = (
        "proj",
    )

    # List of cols
    _base_cols = (
        "i", "solver_iter", "t",
        'CA', 'CY', 'CN', 'CLL', 'CLM', 'CLN',
        'CA_p', 'CY_p', 'CN_p', 'CA_v', 'CY_v', 'CN_v',
        'CA_m', 'CY_m', 'CN_m', 'CLL_p', 'CLM_p', 'CLN_p',
        'CLL_v', 'CLM_v', 'CLN_v', 'CLL_m', 'CLM_v', 'CLN_v',
        'mdot', 'A', 'Ax', 'Ay', 'Az'
    )

    _base_coeffs = (
        'CA', 'CY', 'CN', 'CLL', 'CLM', 'CLN',
        'CA_p', 'CY_p', 'CN_p', 'CA_v', 'CY_v', 'CN_v',
        'CA_m', 'CY_m', 'CN_m', 'CLL_p', 'CLM_p', 'CLN_p',
        'CLL_v', 'CLM_v', 'CLN_v', 'CLL_m', 'CLM_v', 'CLN_v',
        'mdot', 'A', 'Ax', 'Ay', 'Az'
    )

    # Initialization method
    def __init__(self, proj: str, comp: str, **kw):
        r"""Initialization method

        :Versions:
            * 2026-02-03 ``@ddalle``: v1.0
            * 2024-01-24 ``@ddalle``: v2.0; caching upgrade
        """
        # Get the project rootname
        self.proj = proj
        # Pass to parent class
        databook.CaseFM.__init__(self, comp, **kw)

    # Get list of files to read
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
            * 2024-01-24 ``@ddalle``: v1.0
        """
        # Expected name of the component history file
        sources = [
            f"{self.proj}.fomoco",
            "fomoco.out",
            "fomoco.tmp",
        ]
        # Initialize output
        filelist = []
        # Loop through potential files
        for sourcefile in sources:
            # Check if it exists
            if os.path.isfile(sourcefile):
                # Check mod time
                mtime = os.path.getmtime(sourcefile)
                # Only if it's newer than prev file
                if len(filelist) == 0:
                    # No previous file to compare to
                    filelist.append(sourcefile)
                elif mtime > os.path.getmtime(filelist[-1]):
                    # "fomoco.out" is newer than "run.fomoco"
                    filelist.append(sourcefile)
        # Output
        return filelist

    # Read a FOMOCO file
    def readfile(self, fname: str) -> dict:
        r"""Read a FOMOCO output file for one component

        :Call:
            >>> db = fm.readfile(fname)
        :Inputs:
            *fm*: :class:`CaseFM`
                Single-case force & moment iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *db*: :class:`basedata.BaseData`
                Data read from *fname*
        :Versions:
            * 2024-01-24 ``@ddalle``: v1.0
        """
        # Read metadata
        icomp, ncomp, niter = self.read_fomoco_meta(fname, self.comp)
        # Initialize data
        data = np.zeros((niter, 38))
        # Number of bytes in one iteration
        rowsize = 650 * (ncomp - 1)
        # Open the file
        with open(fname, 'rb') as fp:
            # Skip to start of first iteration
            fp.seek(650*icomp + 81)
            # Loop through iterations
            for i in range(niter):
                # Read data
                A = np.fromfile(fp, sep=" ", count=38)
                # Check for iteration
                if len(A) == 38:
                    # Save the data
                    data[i] = A
                # Skip to next iteration
                fp.seek(rowsize + 81, 1)
        # Initialize data for output
        db = basedata.BaseData()
        # Save iterations
        db.save_col(databook.CASE_COL_ITERS, data[:, 0])
        # Time
        db.save_col("wallTime", data[:, 28])
        # Pressure contributions to force
        db.save_col("CA_p", data[:, 6])
        db.save_col("CY_p", data[:, 7])
        db.save_col("CN_p", data[:, 8])
        # Viscous contributions to force
        db.save_col("CA_v", data[:, 9])
        db.save_col("CY_v", data[:, 10])
        db.save_col("CN_v", data[:, 11])
        # Momentum contributions to force
        db.save_col("CA_m", data[:, 12])
        db.save_col("CY_m", data[:, 13])
        db.save_col("CN_m", data[:, 14])
        # Overall force coefficients
        db.save_col("CA", np.sum(data[:, [6, 9, 12]], axis=1))
        db.save_col("CY", np.sum(data[:, [7, 10, 13]], axis=1))
        db.save_col("CN", np.sum(data[:, [8, 11, 14]], axis=1))
        # Pressure contributions to moments
        db.save_col("CLL_p", data[:, 29])
        db.save_col("CLM_p", data[:, 30])
        db.save_col("CLN_p", data[:, 31])
        # Viscous contributions to moments
        db.save_col("CLL_v", data[:, 32])
        db.save_col("CLM_v", data[:, 33])
        db.save_col("CLN_v", data[:, 34])
        # Momentum contributions to moments
        db.save_col("CLL_m", data[:, 35])
        db.save_col("CLM_m", data[:, 36])
        db.save_col("CLN_m", data[:, 37])
        # Moment coefficients
        db.save_col("CLL", np.sum(data[:, [29, 32, 35]], axis=1))
        db.save_col("CLM", np.sum(data[:, [30, 33, 36]], axis=1))
        db.save_col("CLN", np.sum(data[:, [31, 34, 37]], axis=1))
        # Mass flow
        db.save_col("mdot", data[:, 27])
        # Areas
        db.save_col("A", data[:, 2])
        db.save_col("Ax", data[:, 3])
        db.save_col("Ay", data[:, 4])
        db.save_col("Az", data[:, 5])
        # Output
        return db

    # Get stats from a named FOMOCO file
    def read_fomoco_meta(self, fname: str, comp: str):
        r"""Get basic stats about an OVERFLOW fomoco file

        :Call:
            >>> ic, nc, ni = fm.GetFomocoInfo(fname, comp)
        :Inputs:
            *fm*: :class:`CaseFM`
                Force and moment iterative history
            *fname*: :class:`str`
                Name of file to query
            *comp*: :class:`str`
                Name of component to find
        :Outputs:
            *ic*: :class:`int` | ``None``
                Index of component in the list of components
            *nc*: :class:`int` | ``None``
                Number of components
            *ni*: :class:`int`
                Number of iterations
        :Versions:
            * 2016-02-03 ``@ddalle``: v1.0 (GetFomocoInfo)
            * 2024-01-24 ``@ddalle``: v1.1
        """
        # Get list of components
        comps = read_fomoco_comps(fname)
        # Number of components
        nc = len(comps)
        # Check if our component is present
        if comp in comps:
            # Index of the component.
            ic = comps.index(comp)
            # Number of (relevant) iterations
            ni = read_fomoco_niter(fname, nc)
        else:
            # No useful iterations
            ic = 0
            # Number of (relevant) iterations
            ni = 0
        # Output
        return ic, nc, ni


# Residual class
class CaseResid(databook.CaseResid):
    r"""OVERFLOW iterative residual history class

    This class provides an interface to residuals for a given case by reading
    the files ``resid.out``, ``resid.tmp``, ``run.resid``, ``turb.out``,
    ``species.out``, etc.

    :Call:
        >>> hist = CaseResid(proj)
    :Inputs:
        *proj*: :class:`str`
            Project root name
    :Outputs:
        *hist*: :class:`CaseResid`
            Instance of the residual histroy class
    """
    # Default column lists
    _base_cols = (
        "i",
        "solver_iter",
        "L2",
        "LInf",
    )
    _base_coeffs = (
        "L2",
        "LInf",
    )

    # Initialization method
    def __init__(self, proj: str, **kw):
        r"""Initialization method

        :Versions:
            * 2016-02-03 ``@ddalle``: v1.0
            * 2024-01-11 ``@ddalle``: v1.1; DataKit updates
        """
        # Save the prefix
        self.proj = proj
        # Parent initialization
        databook.CaseResid.__init__(self, **kw)

    # Get list of files to read
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
            * 2024-01-24 ``@ddalle``: v1.0
        """
        # Expected name of the component history file
        sources = [
            f"{self.proj}.resid",
            "resid.out",
            "resid.tmp",
        ]
        # Initialize output
        filelist = []
        # Loop through potential files
        for sourcefile in sources:
            # Check if it exists
            if os.path.isfile(sourcefile):
                # Check mod time
                mtime = os.path.getmtime(sourcefile)
                # Only if it's newer than prev file
                if len(filelist) == 0:
                    # No previous file to compare to
                    filelist.append(sourcefile)
                elif mtime > os.path.getmtime(filelist[-1]):
                    # "fomoco.out" is newer than "run.fomoco"
                    filelist.append(sourcefile)
        # Output
        return filelist

    # Read an OVERFLOW resid file
    def readfile(self, fname: str) -> dict:
        r"""Read an OVERFLOW residual history file; create global resid

        :Call:
            >>> db = fm.readfile(fname)
        :Inputs:
            *fm*: :class:`CaseFM`
                Single-case force & moment iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *db*: :class:`basedata.BaseData`
                Data read from *fname*
        :Versions:
            * 2024-01-24 ``@ddalle``: v1.0
        """
        # Number of grids
        ngrid = ReadResidNGrids(fname)
        # Initialize output
        db = basedata.BaseData()
        # Read the file into an array
        A = np.loadtxt(fname, usecols=(1, 2, 3, 13), ndmin=2)
        # Number of iterations
        ncol = 4
        niter = A.shape[0] // ngrid
        # Ad a robust step in case we caught a mid-iteration write
        A = A[:niter*ngrid, :]
        # Reshape data
        A = A.reshape((niter, ngrid, ncol))
        # Save iterationd
        db.save_col(databook.CASE_COL_ITERS, A[:, 0, 0])
        # Add L2's of each grid
        L2 = np.sum(A[:, :, 1]**2, axis=1)
        # Take max of any grid's Linf norm
        Linf = np.max(A[:, :, 2], axis=1)
        # Save them
        db.save_col("L2", L2)
        db.save_col("LInf", Linf)
        # Output
        return db

    # Plot L2 norm
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
            *FigWidth*: :class:`float`
                Figure width
            *FigHeight*: :class:`float`
                Figure height
        :Outputs:
            *h*: :class:`dict`
                Dictionary of figure/plot handles
        :Versions:
            * 2014-11-12 ``@ddalle``: v1.0
            * 2014-12-09 ``@ddalle``: v1.1; move to ``AeroPlot``
            * 2015-02-15 ``@ddalle``: v1.2; move to ``databook.Aero``
            * 2015-03-04 ``@ddalle``: v1.3; add *nStart* and *nLast*
            * 2015-10-21 ``@ddalle``: v1.4; use :func:`PlotResid`
        """
        # Y-label option
        ylbl = kw.get('YLabel', 'L2 Residual')
        # Plot 'L2Resid'
        return self.PlotResid(
            'L2', n=n, nFirst=nFirst, nLast=nLast, YLabel=ylbl, **kw)


# Aerodynamic history class
class DataBook(databook.DataBook):
    r"""DataBook interface for OVERFLOW

    :Call:
        >>> DB = DataBook(x, opts)
    :Inputs:
        *x*: :class:`pyFun.runmatrix.RunMatrix`
            The current pyFun trajectory (i.e. run matrix)
        *opts*: :class:`pyFun.options.Options`
            Global pyFun options instance
    :Outputs:
        *DB*: :class:`DataBook`
            Instance of the pyFun data book class
    """
    _fm_cls = FMDataBook
    _pt_cls = pointsensor.PointSensorGroupDataBook
    _ts_cls = TimeSeriesDataBook
    _prop_cls = PropDataBook
    _pyfunc_cls = PyFuncDataBook

    # Local version of data book
    def _DataBook(self, targ):
        self.Targets[targ] = DataBook(
            self.x, self.opts, RootDir=self.RootDir, targ=targ)

    # Local version of target
    def _TargetDataBook(self, targ):
        self.Targets[targ] = TargetDataBook(
            targ, self.x, self.opts, self.RootDir)

    # Local line load data book read
    def _LineLoadDataBook(self, comp, conf=None, targ=None):
        r"""Version-specific line load reader

        :Versions:
            * 2017-04-18 ``@ddalle``: v1.0
        """
        # Check for target
        if targ is None:
            self.LineLoads[comp] = lineload.LineLoadDataBook(
                comp, self.cntl,
                conf=conf, RootDir=self.RootDir, targ=self.targ)
        else:
            # Read as a specified target.
            ttl = '%s\\%s' % (targ, comp)
            # Get the keys
            topts = self.opts.get_TargetDataBookByName(targ)
            keys = topts.get("Keys", self.x.cols)
            # Read the file.
            self.LineLoads[ttl] = lineload.LineLoadDataBook(
                comp, self.cntl, keys=keys,
                conf=conf, RootDir=self.RootDir, targ=targ)

    # Read point sensor (group)
    def ReadPointSensor(self, name):
        r"""Read a point sensor group if it is not already present

        :Call:
            >>> DB.ReadPointSensor(name)
        :Inputs:
            *DB*: :class:`DataBook`
                Instance of the pycart data book class
            *name*: :class:`str`
                Name of point sensor group
        :Versions:
            * 2015-12-04 ``@ddalle``: (from cape.pycart)
        """
        # Initialize if necessary.
        try:
            self.PointSensors
        except Exception:
            self.PointSensors = {}
        # Initialize the group if necessary
        try:
            self.PointSensors[name]
        except Exception:
            # Safely go to root directory
            fpwd = os.getcwd()
            os.chdir(self.RootDir)
            # Read the point sensor.
            self.PointSensors[name] = self._pt_cls(
                self.x, self.opts, name, RootDir=self.RootDir)
            # Return to starting location
            os.chdir(fpwd)

  # ========
  # Case I/O
  # ========
  # <
    # Current iteration status
    def GetCurrentIter(self):
        r"""Determine iteration number of current folder

        :Call:
            >>> n = DB.GetCurrentIter()
        :Inputs:
            *DB*: :class:`DataBook`
                Instance of data book class
        :Outputs:
            *n*: :class:`int` | ``None``
                Iteration number
        :Versions:
            * 2017-04-13 ``@ddalle``: v1.0
        """
        try:
            return casecntl.GetCurrentIter()
        except Exception:
            return None
