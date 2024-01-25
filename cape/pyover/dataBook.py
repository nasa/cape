r"""
``cape.pyover.dataBook``: DataBook module for OVERFLOW
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
book modules, :mod:`cape.cfdx.dataBook`, :mod:`cape.cfdx.lineLoad`, and
:mod:`cape.cfdx.pointSensor`.  However, some data book types may not be
implemented for all CFD solvers.

:See Also:
    * :mod:`cape.cfdx.dataBook`
    * :mod:`cape.cfdx.lineLoad`
    * :mod:`cape.cfdx.pointSensor`
    * :mod:`cape.pyover.lineLoad`
    * :mod:`cape.options.DataBook`
    * :mod:`cape.pyover.options.DataBook`
"""

# Standard library
import os
from io import IOBase

# Third-party modules
import numpy as np

# Local imports
from . import case
from . import pointSensor
from . import lineLoad
from .. import tri
from .. import fileutils
from ..cfdx import dataBook
from ..attdb.ftypes import basedata


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
            comp = fp.readline().strip()
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
    nGrid = 0
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


# Read the first iteration number from a resid file.
def ReadResidLastIter(fname):
    r"""Read the first iteration number in an OVERFLOW residual file

    :Call:
        >>> nIter = ReadResidLastIter(fname)
        >>> nIter = ReadResidLastIter(f)
    :Inputs:
        *fname*: :class:`str`
            Name of file to query
        *f*: :class:`file`
            Already opened file handle to query
    :Outputs:
        *nIter*: :class:`int`
            Iteration number from last line
    :Versions:
        * 2016-02-04 ``@ddalle``: v1.0
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
    # Go to last line
    fp.seek(-218, 2)
    # Read the second entry from the last line
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


# Aerodynamic history class
class DataBook(dataBook.DataBook):
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
    # Initialize a DBComp object
    def ReadDBComp(self, comp, check=False, lock=False):
        """Initialize data book for one component

        :Call:
            >>> DB.ReadDBComp(comp, check=False, lock=False)
        :Inputs:
            *DB*: :class:`DataBook`
                Instance of the pyCart data book class
            *comp*: :class:`str`
                Name of component
            *check*: ``True`` | {``False``}
                Whether or not to check LOCK status
            *lock*: ``True`` | {``False``}
                If ``True``, wait if the LOCK file exists
        :Versions:
            * 2015-11-10 ``@ddalle``: v1.0
            * 2016-06-27 ``@ddalle``: v1.1; add *targ* keyword
            * 2017-04-13 ``@ddalle``: v1.2; self-contained
        """
        self[comp] = DBComp(
            comp, self.cntl,
            targ=self.targ, check=check, lock=lock)

    # Local version of data book
    def _DataBook(self, targ):
        self.Targets[targ] = DataBook(
            self.x, self.opts, RootDir=self.RootDir, targ=targ)

    # Local version of target
    def _DBTarget(self, targ):
        self.Targets[targ] = DBTarget(targ, self.x, self.opts, self.RootDir)

    # Local line load data book read
    def _DBLineLoad(self, comp, conf=None, targ=None):
        r"""Version-specific line load reader

        :Versions:
            * 2017-04-18 ``@ddalle``: v1.0
        """
        # Check for target
        if targ is None:
            self.LineLoads[comp] = lineLoad.DBLineLoad(
                comp, self.cntl,
                conf=conf, RootDir=self.RootDir, targ=self.targ)
        else:
            # Read as a specified target.
            ttl = '%s\\%s' % (targ, comp)
            # Get the keys
            topts = self.opts.get_DataBookTargetByName(targ)
            keys = topts.get("Keys", self.x.cols)
            # Read the file.
            self.LineLoads[ttl] = lineLoad.DBLineLoad(
                comp, self.cntl, keys=keys,
                conf=conf, RootDir=self.RootDir, targ=targ)

    # Read TriqFM components
    def ReadTriqFM(self, comp, check=False, lock=False):
        r"""Read a TriqFM data book if not already present

        :Call:
            >>> DB.ReadTriqFM(comp)
        :Inputs:
            *DB*: :class:`DataBook`
                Instance of pyOver data book class
            *comp*: :class:`str`
                Name of TriqFM component
            *check*: ``True`` | {``False``}
                Whether or not to check LOCK status
            *lock*: ``True`` | {``False``}
                If ``True``, wait if the LOCK file exists
        :Versions:
            * 2017-03-29 ``@ddalle``: v1.0
        """
        # Initialize if necessary
        try:
            self.TriqFM
        except Exception:
            self.TriqFM = {}
        # Try to access the TriqFM database
        try:
            self.TriqFM[comp]
            # Ensure lock
            if lock:
                self.TriqFM[comp].Lock()
        except Exception:
            # Safely go to root directory
            fpwd = os.getcwd()
            os.chdir(self.RootDir)
            # Read data book
            self.TriqFM[comp] = DBTriqFM(
                self.x, self.opts, comp,
                RootDir=self.RootDir, check=check, lock=lock)
            # Return to starting position
            os.chdir(fpwd)

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
            self.PointSensors[name] = pointSensor.DBPointSensorGroup(
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
            return case.GetCurrentIter()
        except Exception:
            return None

    # Read case residual
    def ReadCaseResid(self):
        r"""Read a :class:`CaseResid` object

        :Call:
            >>> hist = DB.ReadCaseResid()
        :Inputs:
            *DB*: :class:`cape.cfdx.dataBook.DataBook`
                Instance of data book class
        :Outputs:
            *hist*: :class:`CaseResid`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: v1.0
            * 2023-07-10 ``@ddalle``: v1.1; use ``CaseRunner``
        """
        # Get a case runner
        runner = case.CaseRunner()
        # Get the phase number
        k = runner.get_phase()
        # Appropriate prefix
        proj = self.opts.get_Prefix(k)
        # Read CaseResid object from PWD
        return CaseResid(proj)

    # Read case FM history
    def ReadCaseFM(self, comp):
        r"""Read a :class:`CaseFM` object

        :Call:
            >>> fm = DB.ReadCaseFM(comp)
        :Inputs:
            *DB*: :class:`cape.cfdx.dataBook.DataBook`
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
        runner = case.CaseRunner()
        # Get the phase number
        k = runner.get_phase()
        # Appropriate prefix
        proj = self.opts.get_Prefix(k)
        # Read CaseResid object from PWD
        return CaseFM(proj, comp)


# Component data book
class DBComp(dataBook.DBComp):
    r"""Individual component data book

    This class is derived from :class:`cape.cfdx.dataBook.DBBase`.

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
    pass


# Data book target instance
class DBTarget(dataBook.DBTarget):
    pass


# TriqFM data book
class DBTriqFM(dataBook.DBTriqFM):
    r"""Force and moment component extracted from surface triangulation

    :Call:
        >>> DBF = DBTriqFM(x, opts, comp, RootDir=None)
    :Inputs:
        *x*: :class:`cape.runmatrix.RunMatrix`
            RunMatrix/run matrix interface
        *opts*: :class:`cape.options.Options`
            Options interface
        *comp*: :class:`str`
            Name of TriqFM component
        *RootDir*: {``None``} | :class:`st`
            Root directory for the configuration
    :Outputs:
        *DBF*: :class:`DBTriqFM`
            Instance of TriqFM data book
    :Versions:
        * 2017-03-28 ``@ddalle``: v1.0
    """

    # Get file
    def GetTriqFile(self):
        """Get most recent ``triq`` file and its associated iterations

        :Call:
            >>> qpre, fq, n, i0, i1 = DBF.GetTriqFile()
        :Inputs:
            *DBL*: :class:`DBTriqFM`
                Instance of TriqFM data book
        :Outputs:
            *qpre*: {``False``}
                Whether or not to convert file from other format
            *fq*: :class:`str`
                Name of ``q`` file
            *n*: :class:`int`
                Number of iterations included
            *i0*: :class:`int`
                First iteration in the averaging
            *i1*: :class:`int`
                Last iteration in the averaging
        :Versions:
            * 2016-12-19 ``@ddalle``: Added to the module
        """
        # Get Q/X files
        self.fqi = self.opts.get_DataBook_QIn(self.comp)
        self.fxi = self.opts.get_DataBook_XIn(self.comp)
        self.fqo = self.opts.get_DataBook_QOut(self.comp)
        self.fxo = self.opts.get_DataBook_XOut(self.comp)
        # Get properties of triq file
        fq, n, i0, i1 = case.GetQFile(self.fqi)
        # Get the corresponding .triq file name
        ftriq = os.path.join('lineload', 'grid.i.triq')
        # Check for 'q.strt'
        if os.path.isfile(fq):
            # Source file exists
            fsrc = os.path.realpath(fq)
        else:
            # No source just yet
            fsrc = None
        # Check if the TRIQ file exists
        if fsrc and os.path.isfile(ftriq) and os.path.isfile(fsrc):
            # Check modification dates
            if os.path.getmtime(ftriq) < os.path.getmtime(fsrc):
                # 'grid.i.triq' exists, but Q file is newer
                qpre = True
            else:
                # Triq file exists and is up-to-date
                qpre = False
        else:
            # Need to run ``overint`` to get triq file
            qpre = True
        # Output
        return qpre, fq, n, i0, i1

    # Read a Triq file
    def ReadTriq(self, ftriq):
        r"""Read a ``triq`` annotated surface triangulation

        :Call:
            >>> DBF.ReadTriq(ftriq)
        :Inputs:
            *DBF*: :class:`DBTriqFM`
                Instance of TriqFM data book
            *ftriq*: :class:`str`
                Name of ``triq`` file
        :Versions:
            * 2017-03-29 ``@ddalle``: v1.0
        """
        # Check if the configuration file exists
        if os.path.isfile(self.conf):
            # Use that file (explicitly defined)
            fcfg = self.conf
        else:
            # Check for a mixsur file
            fmixsur = self.opts.get_DataBook_mixsur(self.comp)
            fusurp = self.opts.get_DataBook_usurp(self.comp)
            # De-none
            if fmixsur is None:
                fmixsur = ''
            if fusurp is None:
                fusurp = ''
            # Make absolute
            if not os.path.isabs(fmixsur):
                fmixsur = os.path.join(self.RootDir, fmixsur)
            if not os.path.isabs(fusurp):
                fusurp = os.path.join(self.RootDir, fusurp)
            # Read them
            if os.path.isfile(fusurp):
                # USURP file specified; overrides mixsur
                fcfg = fusurp
            elif os.path.isfile(fmixsur):
                # Use MIXSUR file
                fcfg = fmixsur
            else:
                # No config file... probably won't turn out well
                fcfg = None
        # Read from lineload/ folder
        ftriq = os.path.join('lineload', 'grid.i.triq')
        # Read using :mod:`cape`
        self.triq = tri.Triq(ftriq, c=fcfg)

    # Preprocess triq file (convert from PLT)
    def PreprocessTriq(self, fq, **kw):
        r"""Perform any necessary preprocessing to create ``triq`` file

        :Call:
            >>> ftriq = DBF.PreprocessTriq(fq, qpbs=False, f=None)
        :Inputs:
            *DBL*: :class:`DBTriqFM`
                TriqFM data book
            *ftriq*: :class:`str`
                Name of q file
            *qpbs*: ``True`` | {``False``}
                Whether or not to create a script and submit it
            *f*: {``None``} | :class:`file`
                File handle if writing PBS script
        :Versions:
            * 2016-12-20 ``@ddalle``: v1.0
            * 2016-12-21 ``@ddalle``: Added PBS
        """
        # Create lineload folder if necessary
        if not os.path.isdir('lineload'):
            self.opts.mkdir('lineload')
        # Enter line load folder
        os.chdir('lineload')
        # Add '..' to the path
        fq = os.path.join('..', fq)
        # Call local function
        lineLoad.PreprocessTriqOverflow(self, fq)


# Force/moment history
class CaseFM(dataBook.CaseFM):
    r"""Force and moment iterative histories

    This class contains methods for reading data about an the history of
    an individual component for a single case. It reads the Tecplot
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
        dataBook.CaseFM.__init__(self, comp, **kw)

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
        db.save_col(dataBook.CASE_COL_ITERS, data[:, 0])
        # Time
        db.save_col(dataBook.CASE_COL_TIME, data[:, 28])
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
class CaseResid(dataBook.CaseResid):
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
        dataBook.CaseResid.__init__(self, **kw)

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
        # Read the file into an array
        A = np.loadtxt(fname, usecols=(1, 2, 3, 13), ndmin=2)
        # Number of iterations
        ncol = 4
        niter = A.shape[0] // ngrid
        # Reshape data
        A = np.reshape((niter, ngrid, ncol))
        # Get global residuals
        if c == "L2":
            # Get weighted sum
            L = np.sum(B[I, :, 1]*B[I, :, 2]**2, axis=1)
            # Total grid points in each iteration
            N = np.sum(B[I, :, 2], axis=1)
            # Divide by number of grid points, and take square root
            L = np.sqrt(L/N)
            # Append to data
            self.save_col("L2", np.hstack((self["L2"], L)))
        else:
            # Get the maximum value
            L = np.max(B[I, :, 1], axis=1)
            # Append to data
            self.save_col("LInf", np.hstack((self["LInf"], L)))

    # Read entire global residual history
    def ReadGlobalL2(self, grid=None):
        r"""Read entire global L2 history

        The file ``history.L2.dat`` is also updated.

        :Call:
            >>> hist.ReadGlobalL2(grid=None)
        :Inputs:
            *hist*: :class:`CaseResid`
                Iterative residual history class
            *grid*: {``None``} | :class:`int` | :class:`str`
                If used, read only one grid
        :Versions:
            * 2016-02-04 ``@ddalle``: v1.0
            * 2017-04-19 ``@ddalle``: v1.1; add *grid* option
            * 2024-01-11 ``@ddalle``: v1.2; DataKit updates
        """
        # Read the global history file
        if grid is None:
            # Read
            iters, L2 = self.ReadGlobalHist('history.L2.dat')
            # Save
            self.save_col("i", iters)
            self.save_col("L2", L2)
        # OVERFLOW file names
        frun = '%s.resid' % self.proj
        fout = 'resid.out'
        ftmp = 'resid.tmp'
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="L2", grid=grid)
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="L2", grid=grid)
        # Write the updated history (tmp file not safe to write here)
        if grid is None:
            self.WriteGlobalHist('history.L2.dat', iters, L2)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="L2", grid=grid)

    # Read entire L-inf residual
    def ReadGlobalLInf(self, grid=None):
        r"""Read entire L-infinity norm history

        The file ``history.LInf.dat`` is also updated

        :Call:
            >>> hist.ReadGlobalLInf(grid=None)
        :Inputs:
            *hist*: :class:`CaseResid`
                Iterative residual history class
            *grid*: {``None``} | :class:`int` | :class:`str`
                If used, read only one grid
        :Versions:
            * 2016-02-06 ``@ddalle``: v1.0
            * 2017-04-19 ``@ddalle``: v1.1; add *grid* option
            * 2024-01-11 ``@ddalle``: v1.2; DataKit updates
        """
        # Read the global history file
        if grid is None:
            iters, LInf = self.ReadGlobalHist('histroy.LInf.dat')
            self.save_col("i", iters)
            self.save_col("LInf", LInf)
        # OVERFLOW file names
        frun = '%s.resid' % self.proj
        fout = 'resid.out'
        ftmp = 'resid.tmp'
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="LInf", grid=grid)
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="LInf", grid=grid)
        # Write the updated history (tmp file not safe to write here)
        if grid is None:
            self.WriteGlobalHist('history.LInf.dat', iters, LInf)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="LInf", grid=grid)

    # Read turbulence L2 residual
    def ReadTurbResidL2(self, grid=None):
        r"""Read the entire L2 norm of the turbulence residuals

        The file ``history.turb.L2.dat`` is also updated

        :Call:
            >>> hist.ReadTurbResidL2(grid=None)
        :Inputs:
            *hist*: :class:`CaseResid`
                Iterative residual history class
            *grid*: {``None``} | :class:`int` | :class:`str`
                If used, read only one grid
        :Versions:
            * 2016-02-06 ``@ddalle``: v1.0
            * 2017-04-19 ``@ddalle``: v1.1; add *grid* option
            * 2024-01-11 ``@ddalle``: v1.2; DataKit updates
        """
        # Read the global history file
        if grid is None:
            iters, L2turb = self.ReadGlobalHist('history.turb.L2.dat')
            self.save_col("i", iters)
            self.save_col("L2turb", L2turb)
        # OVERFLOW file names
        frun = '%s.turb' % self.proj
        fout = 'turb.out'
        ftmp = 'turb.tmp'
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="L2turb", grid=grid)
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="L2turb", grid=grid)
        # Write the updated history (tmp file not safe to write here)
        if grid is None:
            self.WriteGlobalHist('history.turb.L2.dat', iters, L2turb)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="L2turb", grid=grid)

    # Read turbulence LInf residual
    def ReadTurbResidLInf(self, grid=None):
        r"""Read the global L-infinity norm of the turbulence residuals

        The file ``history.turb.LInf.dat`` is also updated

        :Call:
            >>> hist.ReadTurbResidLInf()
        :Inputs:
            *hist*: :class:`CaseResid`
                Iterative residual history class
            *grid*: {``None``} | :class:`int` | :class:`str`
                If used, read only one grid
        :Versions:
            * 2016-02-06 ``@ddalle``: v1.0
            * 2017-04-19 ``@ddalle``: v1.1; add *grid* option
            * 2024-01-11 ``@ddalle``: v1.2; DataKit updates
        """
        # Read the global history file
        if grid is None:
            iters, LInfturb = self.ReadglobalHist('history.turb.LInf.dat')
            self.save_col("i", iters)
            self.save_col("LInfturb", LInfturb)
        # OVERFLOW file names
        frun = '%s.turb' % self.proj
        fout = 'turb.out'
        ftmp = 'turb.tmp'
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="LInfturb", grid=grid)
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="LInfturb", grid=grid)
        # Write the updated history (tmp file not safe to write here)
        if grid is None:
            self.WriteGlobalHist('history.turb.LInf.dat', iters, LInfturb)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="LInfturb", grid=grid)

    # Read species L2 residual
    def ReadSpeciesResidL2(self, grid=None):
        r"""Read the global L2 norm of the species equations

        The file ``history.species.L2.dat`` is also updated

        :Call:
            >>> hist.ReadSpeciesResidL2(grid=None)
        :Inputs:
            *hist*: :class:`CaseResid`
                Iterative residual history class
            *grid*: {``None``} | :class:`int` | :class:`str`
                If used, read only one grid
        :Versions:
            * 2016-02-06 ``@ddalle``: v1.0
            * 2017-04-19 ``@ddalle``: v1.1; add *grid* option
            * 2024-01-11 ``@ddalle``: v1.2; DataKit updates
        """
        # Read the global history file
        if grid is None:
            iters, L2 = self.ReadglobalHist('history.species.L2.dat')
            self.save_col("i", iters)
            self.save_col("L2", L2)
        # OVERFLOW file names
        frun = '%.species' % self.proj
        fout = 'species.out'
        ftmp = 'species.tmp'
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="L2", grid=grid)
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="L2", grid=grid)
        # Write the updated history (tmp file not safe to write here)
        if grid is None:
            self.WriteGlobalHist('history.species.L2.dat', iters, L2)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="L2", grid=grid)

    # Read species LInf residual
    def ReadSpeciesResidLInf(self, grid=None):
        r"""Read the global L-infinity norm of the species equations

        The file ``history.species.LInf.dat`` is also updated

        :Call:
            >>> hist.ReadSpeciesResidLInf(grid=None)
        :Inputs:
            *hist*: :class:`CaseResid`
                Iterative residual history class
            *grid*: {``None``} | :class:`int` | :class:`str`
                If used, read only one grid
        :Versions:
            * 2016-02-06 ``@ddalle``: v1.0
            * 2017-04-19 ``@ddalle``: v1.1; add *grid* option
            * 2024-01-11 ``@ddalle``: v1.2; DataKit updates
        """
        # Read the global history file
        if grid is None:
            iters, LInf = self.ReadglobalHist('history.species.LInf.dat')
            self.save_col("i", iters)
            self.save_col("LInf", LInf)
        # OVERFLOW file names
        frun = '%.species' % self.proj
        fout = 'species.out'
        ftmp = 'species.tmp'
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="LInf", grid=grid)
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="LInf", grid=grid)
        # Write the updated history (tmp file not safe to write here)
        if grid is None:
            self.WriteGlobalHist('history.species.LInf.dat', iters, LInf)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="LInf", grid=grid)

    # Read a consolidated history file
    def ReadGlobalHist(self, fname: str):
        r"""Read a condensed global residual file for faster read times

        :Call:
            >>> i, L = hist.ReadGlobalHist(fname)
        :Inputs:
            *hist*: :class:`CaseResid`
                Iterative residual history class
        :Outputs:
            *i*: :class:`numpy.ndarray`\ [:class:`int`]
                Iterations at which residuals are recorded
            *L*: :class:`numpy.ndarray`\ [:class:`float`]
                Residual at each iteration
        :Versions:
            * 2016-02-04 ``@ddalle``: v1.0
        """
        # Try to read the file
        try:
            # Read the file.
            A = np.loadtxt(fname)
            # Split into columns
            return A[:, 0], A[:, 1]
        except Exception:
            # Reading file failed
            return np.zeros(0), np.zeros(0)

    # Write a consolidated history file
    def WriteGlobalHist(self, fname, i, L, n=None):
        r"""Write a condensed global residual file for faster read times

        :Call:
            >>> hist.WriteGlobalHist(fname, i, L, n=None)
        :Inputs:
            *hist*: :class:`CaseResid`
                Iterative residual history class
            *i*: :class:`np.ndarray` (:class:`float` | :class:`int`)
                Vector of iteration numbers
            *L*: :class:`np.ndarray`\ [:class:`float`]
                Vector of residuals to write
            *n*: :class:`int` | ``None``
                Last iteration already written to file.
        :Versions:
            * 2016-02-04 ``@ddalle``: v1.0
        """
        # Default number of lines to skip
        if n is None:
            # Query the file.
            if os.path.isfile(fname):
                try:
                    # Read the last line of the file
                    line = fileutils .tail(fname)
                    # Get the iteration number
                    n = float(line.split()[0])
                except Exception:
                    # File exists but has some issues
                    n = 0
            else:
                # Start at the beginning of the array
                n = 0
        # Find the index of the first iteration greater than *n*
        I = np.where(i > n)[0]
        # If no hits, nothing to write
        if len(I) == 0:
            return
        # Index to start at
        istart = I[0]
        # Append to the file
        with open(fname, 'a') as fp:
            # Loop through the lines
            for j in range(istart, len(i)):
                # Write iteration
                fp.write('%8i %14.7E\n' % (i[j], L[j]))

    # Read a global residual file
    def ReadResidGlobal(self, fname, coeff="L2", n=None, grid=None):
        r"""Read a global residual from one file

        :Call:
            >>> i, L2 = hist.ReadResidGlobal(fname, coeff="L2", **kw)
            >>> i, LInf = hist.ReadResidGlobal(fname, coeff="LInf", **kw)
        :Inputs:
            *hist*: :class:`CaseResid`
                Iterative residual history class
            *fname*: :class:`str`
                Name of file to process
            *coeff*: :class:`str`
                Name of coefficient to read
            *n*: {``None``} | :class:`int`
                Number of last iteration that's already processed
            *grid*: {``None``} | :class:`int` | :class:`str`
                If used, read only one grid
        :Outputs:
            *i*: :class:`np.ndarray`\ [:class:`float`]
                Array of iteration numbers
            *L2*: :class:`np.ndarray`\ [:class:`float`]
                Array of weighted global L2 norms
            *LInf*: :class:`np.ndarray`\ [:class:`float`]
                Array of global L-infinity norms
        :Versions:
            * 2016-02-04 ``@ddalle``: v1.0
            * 2017-04-19 ``@ddalle``: v1.1; add *grid* option
            * 2024-01-11 ``@ddalle``: v1.2; DataKit updates
        """
        # Check for individual grid
        if grid is not None:
            # Pass to individual grid reader
            iL = self.ReadResidGrid(fname, grid=grid, coeff=coeff, n=n)
            # Quit.
            return iL
        # Check for the file
        if not os.path.isfile(fname):
            return
        # First iteration
        i0 = ReadResidFirstIter(fname)
        # Number of iterations
        nIter = ReadResidNIter(fname)
        self.nIter = nIter
        # Number of grids
        nGrid = ReadResidNGrids(fname)
        # Process current iteration number
        if n is None:
            # Use last known iteration
            if len(self["i"]) == 0:
                # No iterations
                n = 0
            else:
                # Use last current iter
                n = int(np.max(self["i"]))
        # Number of iterations to skip
        nIterSkip = max(0, n-i0+1)
        # Skip *nGrid* rows for each iteration
        nSkip = int(nIterSkip * nGrid)
        # Number of iterations to be read
        nIterRead = nIter - nIterSkip
        # Check for something to read
        if nIterRead <= 0:
            return np.zeros(0), np.zeros(0)
        # Process columns to read
        if coeff.lower() == "linf":
            # Read the iter, L-infinity norm
            cols = (1, 3)
            nc = 2
            # Coefficient
            c = 'LInf'
        else:
            # Read the iter, L2 norm, nPts
            cols = (1, 2, 13)
            nc = 3
            # Field name
            c = 'L2'
        # Read the file
        A = np.loadtxt(fname, skiprows=nSkip, usecols=cols)
        # Reshape the data
        B = np.reshape(A[:nIterRead*nGrid, :], (nIterRead, nGrid, nc))
        # Get iterations
        i = B[:, 0, 0]
        # Filter iterations greater than *n*
        I = i > n
        i = i[I]
        # Exit if no iterations
        if len(i) == 0:
            return
        # Get global residuals
        if c == "L2":
            # Get weighted sum
            L = np.sum(B[I, :, 1]*B[I, :, 2]**2, axis=1)
            # Total grid points in each iteration
            N = np.sum(B[I, :, 2], axis=1)
            # Divide by number of grid points, and take square root
            L = np.sqrt(L/N)
            # Append to data
            self.save_col("L2", np.hstack((self["L2"], L)))
        else:
            # Get the maximum value
            L = np.max(B[I, :, 1], axis=1)
            # Append to data
            self.save_col("LInf", np.hstack((self["LInf"], L)))
        # Check for issues
        if np.any(np.diff(i) < 0):
            # Warning
            print(
                f"  Warning: file {fname}' contains non-ascending iterations")
        # Append to data
        self.save_col('i', np.hstack((self["i"], i)))
        # Output
        return i, L

    # Read a global residual file
    def ReadResidGrid(self, fname, grid=None, coeff="L2", n=None):
        r"""Read a global residual from one file

        :Call:
            >>> i, L = hist.ReadResidGrid(fname, grid, coeff="L2", **kw)
        :Inputs:
            *hist*: :class:`CaseResid`
                Iterative residual history class
            *fname*: :class:`str`
                Name of file to process
            *grid*: {``None``} | :class:`int` | :class:`str`
                If used, read history of a single grid
            *coeff*: {``"L2"``} | :class:`str`
                Name of coefficient to read
            *n*: :class:`int` | ``None``
                Number of last iteration that's already processed
        :Outputs:
            *i*: :class:`np.ndarray`\ [:class:`float`]
                Array of iteration numbers
            *L*: :class:`np.ndarray`\ [:class:`float`]
                Array of weighted global residual history
        :Versions:
            * 2017-04-19 ``@ddalle``: v1.0
            * 2024-01-11 ``@ddalle``: v1.1; DataKit updates
        """
        # Check for the file
        if not os.path.isfile(fname):
            return None, None
        # First iteration
        i0 = ReadResidFirstIter(fname)
        # Number of iterations
        nIter = ReadResidNIter(fname)
        self.nIter = nIter
        # Number of grids
        nGrid = ReadResidNGrids(fname)
        # Process current iteration number
        if n is None:
            # Use last known iteration
            if len(self["i"]) == 0:
                # No iterations
                n = 0
            else:
                # Use last current iter
                n = int(np.max(self["i"]))
        # Individual grid
        if grid is None:
            # Read all grids
            kGrid = nGrid
        elif isinstance(grid, str):
            # Figure out grid number
            grids = ReadResidGrids(fname)
            # Check presence
            if grid not in grids:
                raise ValueError(f"Could not find grid '{grid}'")
            # Get index
            iGrid = grids.index(grid)
            kGrid = 1
        elif grid < 0:
            # Read from the back
            iGrid = nGrid + grid
            kGrid = 1
        else:
            # Read from the front (zero-based)
            iGrid = grid - 1
            kGrid = 1
        # Grid range
        KGrid = np.arange(kGrid)
        # Number of iterations to skip
        nIterSkip = int(max(0, n-i0+1))
        # Skip *nGrid* rows for each iteration
        nSkip = int(nIterSkip * nGrid)
        # Number of iterations to be read
        nIterRead = nIter - nIterSkip
        # Check for something to read
        if nIterRead <= 0:
            return np.array([]), np.array([])
        # Process columns to read
        if coeff.lower() == "linf":
            # Read the iter, L-infinity norm
            cols = (1, 3)
            nc = 2
            # Coefficient
            c = 'LInf'
        else:
            # Read the iter, L2 norm, nPts
            cols = (1, 2, 13)
            nc = 3
            # Field name
            c = 'L2'
        # Initialize matrix
        B = np.zeros((nIterRead, kGrid, nc))
        # Open the file
        with open(fname, 'r') as fp:
            # Skip desired number of rows
            fp.seek(nSkip*218)
            # Check if we should skip to grid *grid*
            if grid is not None:
                fp.seek(iGrid*218 + fp.tell())
            # Loop through iterations
            for j in np.arange(nIterRead):
                # Loop through grids
                for k in KGrid:
                    # Read data
                    bjk = np.fromfile(fp, sep=" ", count=-1)
                    # Save it
                    B[j, k, :] = bjk[cols]
                    # Skip over the string
                    fp.seek(26 + fp.tell())
                # Skip rows if appropriate
                if grid is not None:
                    fp.seek((nGrid-1)*218 + fp.tell())
        # Get iterations
        i = B[:, 0, 0]
        # Filter iterations greater than *n*
        I = i > n
        i = i[I]
        # Exit if no iterations
        if len(i) == 0:
            return
        # Get global residuals
        if c == "L2":
            # Get weighted sum
            L = np.sum(B[I, :, 1]*B[I, :, 2]**2, axis=1)
            # Total grid points in each iteration
            N = np.sum(B[I, :, 2], axis=1)
            # Divide by number of grid points, and take square root
            L = np.sqrt(L/N)
            # Append to data
            self.save_col("L2", np.hstack((self["L2"], L)))
        else:
            # Get the maximum value
            L = np.max(B[I, :, 1], axis=1)
            # Append to data
            self.save_col("LInf", np.hstack((self["LInf"], L)))
        # Check for issues
        if np.any(np.diff(i) < 0):
            # Warning
            print(
                f"  Warning: file '{fname}' contains non-ascending iterations")
        # Append to data
        self.save_col("i", np.hstack((self["i"], i)))
        # Output
        return i, L

    # Plot L2 norm
    def PlotL2(self, n=None, nFirst=None, nLast=None, **kw):
        r"""Plot the L2 residual

        :Call:
            >>> h = hist.PlotL2(n=None, nFirst=None, nLast=None, **kw)
        :Inputs:
            *hist*: :class:`cape.cfdx.dataBook.CaseResid`
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
            * 2015-02-15 ``@ddalle``: v1.2; move to ``dataBook.Aero``
            * 2015-03-04 ``@ddalle``: v1.3; add *nStart* and *nLast*
            * 2015-10-21 ``@ddalle``: v1.4; use :func:`PlotResid`
        """
        # Y-label option
        ylbl = kw.get('YLabel', 'L2 Residual')
        # Plot 'L2Resid'
        return self.PlotResid(
            'L2', n=n, nFirst=nFirst, nLast=nLast, YLabel=ylbl)

