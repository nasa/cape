# -*- coding: utf-8 -*-
r"""
:mod:`cape.pyover.dataBook`: pyOver data book module 
=====================================================

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
import re
import shutil
from datetime import datetime

# Third-party modules
import numpy as np

# Local imports
from . import util
from . import bin
from . import case
from . import pointSensor
from . import lineLoad

# Template module
import cape.cfdx.dataBook
import cape.tri


# Placeholder variables for plotting functions.
plt = 0

# Radian -> degree conversion
deg = np.pi / 180.0

# Dedicated function to load Matplotlib only when needed.
def ImportPyPlot():
    """Import :mod:`matplotlib.pyplot` if not loaded
    
    :Call:
        >>> pyOver.dataBook.ImportPyPlot()
    :Versions:
        * 2014-12-27 ``@ddalle``: First version
        * 2016-01-02 ``@ddalle``: Copied from pyCart
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
        import matplotlib.transforms as tform
        from matplotlib.text import Text
# def ImportPyPlot

# Read component names from a fomoco file
def ReadFomocoComps(fname):
    """Get list of components in an OVERFLOW fomoco file
    
    :Call:
        >>> comps = pyOver.dataBook.ReadFomocoComps(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of the file to read
    :Outputs:
        *comps*: :class:`list`\ [:class:`str`]
            List of components
    :Versions:
        * 2016-02-03 ``@ddalle``: First version
    """
    # Initialize components
    comps = []
    # Open the file
    f = open(fname, 'r')
    # Read the first line and first component.
    comp = f.readline().strip()
    # Loop until a component repeats
    while comp not in comps:
        # Check for empty line
        if comp == "":
            break
        # Add the component
        comps.append(comp)
        # Move to the next component
        f.seek(569 + f.tell())
        # Read the next component.
        comp = f.readline().strip()
    # Close the file
    f.close()
    # Output
    return comps
    
# Read basic stats from a fomoco file
def ReadFomocoNIter(fname, nComp=None):
    """Get number of iterations in an OVERFLOW fomoco file
    
    :Call:
        >>> nIter = pyOver.dataBook.ReadFomocoNIter(fname)
        >>> nIter = pyOver.dataBook.ReadFomocoNIter(fname, nComp)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *nComp*: :class:`int` | ``None``
            Number of components in each record
    :Outputs:
        *nIter*: :class:`int`
            Number of iterations in the file
    :Versions:
        * 2016-02-03 ``@ddalle``: First version
    """
    # If no number of comps, get list
    if nComp is None:
        # Read component list
        comps = ReadFomocoComps(fname)
        # Number of components
        nComp = len(comps)
    # Open file to get number of iterations
    f = open(fname)
    # Go to end of file to get length of file
    f.seek(0, 2)
    # Position at EOF
    L = f.tell()
    # Close the file.
    f.close()
    # Save number of iterations
    return int(np.ceil(L / (nComp*650.0)))
# def ReadFomoco

# Read grid names from a resid file
def ReadResidGrids(fname):
    """Get list of grids in an OVERFLOW residual file
    
    :Call:
        >>> grids = pyOver.dataBook.ReadResidGrids(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *grids*: :class:`list`\ [:class:`str`]
            List of grids
    :Versions:
        * 2016-02-04 ``@ddalle``: First version
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
    while len(comps)==0 or comp!=comps[0]:
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
    """Get number of grids from an OVERFLOW residual file
    
    :Call:
        >>> nGrid = pyOver.dataBook.ReadResidNGrids(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *nGrid*: :class:`int`
            Number of grids
    :Versions:
        * 2016-02-04 ``@ddalle``: First version
    """
    # Initialize number of grids
    nGrid = 0
    # Open the file
    f = open(fname, 'r')
    # Read the first grid number
    iGrid = int(f.readline().split()[0])
    # Loop until grid number decreases
    while iGrid > nGrid:
        # Update grid count
        nGrid += 1
        # Read the next line.
        line = f.readline().split()
        # Check for EndOfFile
        if len(line) == 0:
            break
        # Read the next grid number.
        iGrid = int(line[0])
    # Close the file
    f.close()
    # Output
    return nGrid
    
# Read the first iteration number from a resid file.
def ReadResidFirstIter(fname):
    """Read the first iteration number in an OVERFLOW residual file
    
    :Call:
        >>> iIter = pyOver.dataBook.ReadResidFirstIter(fname)
        >>> iIter = pyOver.dataBook.ReadResidFirstIter(f)
    :Inputs:
        *fname*: :class:`str`
            Name of file to query
        *f*: :class:`file`
            Already opened file handle to query
    :Outputs:
        *iIter*: :class:`int`
            Iteration number from first line
    :Versions:
        * 2016-02-04 ``@ddalle``: First version
    """
    # Check input type
    if type(fname).__name__ == "file":
        # Already a file.
        f = fname
        # Check if it's open already
        qf = True
        # Get current location
        ft = f.tell()
    else:
        # Open the file.
        f = open(fname, 'r')
        # Not open
        qf = False
    # Read the second entry from the first line
    iIter = int(f.readline().split()[1])
    # Close the file.
    if qf:
        # Return to original location
        f.seek(ft)
    else:
        # Close the file
        f.close()
    # Output
    return iIter
    
# Read the first iteration number from a resid file.
def ReadResidLastIter(fname):
    """Read the first iteration number in an OVERFLOW residual file
    
    :Call:
        >>> nIter = pyOver.dataBook.ReadResidLastIter(fname)
        >>> nIter = pyOver.dataBook.ReadResidLastIter(f)
    :Inputs:
        *fname*: :class:`str`
            Name of file to query
        *f*: :class:`file`
            Already opened file handle to query
    :Outputs:
        *nIter*: :class:`int`
            Iteration number from last line
    :Versions:
        * 2016-02-04 ``@ddalle``: First version
    """
    # Check input type
    if type(fname).__name__ == "file":
        # Already a file.
        f = fname
        # Check if it's open already
        qf = True
        # Get current location
        ft = f.tell()
    else:
        # Open the file.
        f = open(fname, 'r')
        # Not open
        qf = False
    # Go to last line
    f.seek(-218, 2)
    # Read the second entry from the last line
    iIter = int(f.readline().split()[1])
    # Close the file.
    if qf:
        # Return to original location
        f.seek(ft)
    else:
        # Close the file
        f.close()
    # Output
    return iIter
    
# Get number of iterations from a resid file
def ReadResidNIter(fname):
    r"""Get number of iterations in an OVERFLOW residual file
    
    :Call:
        >>> nIter = pyOver.dataBook.ReadResidNIter(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to query
    :Outputs:
        *nIter*: :class:`int`
            Number of iterations
    :Versions:
        * 2016-02-04 ``@ddalle``: Version 1.0
        * 2022-01-09 ``@ddalle``: Version 1.1; Python 3 int division
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
# def ReadResid

# Aerodynamic history class
class DataBook(cape.cfdx.dataBook.DataBook):
    """
    This class provides an interface to the data book for a given CFD run
    matrix.
    
    :Call:
        >>> DB = pyFun.dataBook.DataBook(x, opts)
    :Inputs:
        *x*: :class:`pyFun.runmatrix.RunMatrix`
            The current pyFun trajectory (i.e. run matrix)
        *opts*: :class:`pyFun.options.Options`
            Global pyFun options instance
    :Outputs:
        *DB*: :class:`pyFun.dataBook.DataBook`
            Instance of the pyFun data book class
    :Versions:
        * 2015-10-20 ``@ddalle``: Started
    """
    # Initialize a DBComp object
    def ReadDBComp(self, comp, check=False, lock=False):
        """Initialize data book for one component
        
        :Call:
            >>> DB.ReadDBComp(comp, x, opts, check=False, lock=False)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *comp*: :class:`str`
                Name of component
            *check*: ``True`` | {``False``}
                Whether or not to check LOCK status
            *lock*: ``True`` | {``False``}
                If ``True``, wait if the LOCK file exists
        :Versions:
            * 2015-11-10 ``@ddalle``: First version
            * 2016-06-27 ``@ddalle``: Added *targ* keyword
            * 2017-04-13 ``@ddalle``: Self-contained and renamed
        """
        self[comp] = DBComp(comp, self.x, self.opts,
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
        """Version-specific line load reader
        
        :Versions:
            * 2017-04-18 ``@ddalle``: First version
        """
        # Check for target
        if targ is None:
            self.LineLoads[comp] = lineLoad.DBLineLoad(
                self.x, self.opts, comp,
                conf=conf, RootDir=self.RootDir, targ=self.targ)
        else:
            # Read as a specified target.
            ttl = '%s\\%s' % (targ, comp)
            # Get the keys
            topts = self.opts.get_DataBookTargetByName(targ)
            keys = topts.get("Keys", self.x.cols)
            # Read the file.
            self.LineLoads[ttl] = lineLoad.DBLineLoad(
                self.x, self.opts, comp, keys=keys,
                conf=conf, RootDir=self.RootDir, targ=targ)
    
    # Read TriqFM components
    def ReadTriqFM(self, comp, check=False, lock=False):
        """Read a TriqFM data book if not already present
        
        :Call:
            >>> DB.ReadTriqFM(comp)
        :Inputs:
            *DB*: :class:`pyOver.dataBook.DataBook`
                Instance of pyOver data book class
            *comp*: :class:`str`
                Name of TriqFM component
            *check*: ``True`` | {``False``}
                Whether or not to check LOCK status
            *lock*: ``True`` | {``False``}
                If ``True``, wait if the LOCK file exists
        :Versions:
            * 2017-03-29 ``@ddalle``: First version
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
            self.TriqFM[comp] = DBTriqFM(self.x, self.opts, comp,
                RootDir=self.RootDir, check=check, lock=lock)
            # Return to starting position
            os.chdir(fpwd)
    
    # Read point sensor (group)
    def ReadPointSensor(self, name):
        """Read a point sensor group if it is not already present
        
        :Call:
            >>> DB.ReadPointSensor(name)
        :Inputs:
            *DB*: :class:`pyOver.dataBook.DataBook`
                Instance of the pycart data book class
            *name*: :class:`str`
                Name of point sensor group
        :Versions:
            * 2015-12-04 ``@ddalle``: Copied from pyCart
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
        """Determine iteration number of current folder
        
        :Call:
            >>> n = DB.GetCurrentIter()
        :Inputs:
            *DB*: :class:`pyOver.dataBook.DataBook`
                Instance of data book class
        :Outputs:
            *n*: :class:`int` | ``None``
                Iteration number
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        try:
            return case.GetCurrentIter()
        except Exception:
            return None
        
    # Read case residual
    def ReadCaseResid(self):
        """Read a :class:`CaseResid` object
        
        :Call:
            >>> H = DB.ReadCaseResid()
        :Inputs:
            *DB*: :class:`cape.cfdx.dataBook.DataBook`
                Instance of data book class
        :Outputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Get the phase number
        rc = case.ReadCaseJSON()
        k = case.GetPhaseNumber(rc)
        # Appropriate prefix
        proj = self.opts.get_Prefix(k)
        # Read CaseResid object from PWD
        return CaseResid(proj)
        
    # Read case FM history
    def ReadCaseFM(self, comp):
        """Read a :class:`CaseFM` object
        
        :Call:
            >>> FM = DB.ReadCaseFM(comp)
        :Inputs:
            *DB*: :class:`cape.cfdx.dataBook.DataBook`
                Instance of data book class
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *FM*: :class:`pyOver.dataBook.CaseFM`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Get the phase number
        rc = case.ReadCaseJSON()
        k = case.GetPhaseNumber(rc)
        # Appropriate prefix
        proj = self.opts.get_Prefix(k)
        # Read CaseResid object from PWD
        return CaseFM(proj, comp)
  # >
    
# class DataBook

# Component data book
class DBComp(cape.cfdx.dataBook.DBComp):
    """Individual component data book
    
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
        *DBc*: :class:`pyOver.dataBook.DBComp`
            An individual component data book
    :Versions:
        * 2016-09-15 ``@ddalle``: First version
    """
    pass
# class DBComp


# Data book target instance
class DBTarget(cape.cfdx.dataBook.DBTarget):
    """
    Class to handle data from data book target files.  There are more
    constraints on target files than the files that data book creates, and raw
    data books created by pyCart are not valid target files.
    
    :Call:
        >>> DBT = DBTarget(targ, x, opts)
    :Inputs:
        *targ*: :class:`pyOver.options.DataBook.DBTarget`
            Instance of a target source options interface
        *x*: :class:`pyOver.runmatrix.RunMatrix`
            Run matrix interface
        *opts*: :class:`pyOver.options.Options`
            Global pyCart options instance to determine which fields are useful
    :Outputs:
        *DBT*: :class:`pyOver.dataBook.DBTarget`
            Instance of the pyCart data book target data carrier
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
    """
    
    pass
# class DBTarget


# TriqFM data book
class DBTriqFM(cape.cfdx.dataBook.DBTriqFM):
    """Force and moment component extracted from surface triangulation
    
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
        *DBF*: :class:`pyFun.dataBook.DBTriqFM`
            Instance of TriqFM data book
    :Versions:
        * 2017-03-28 ``@ddalle``: First version
    """
    
    # Get file
    def GetTriqFile(self):
        """Get most recent ``triq`` file and its associated iterations
        
        :Call:
            >>> qpre, fq, n, i0, i1 = DBF.GetTriqFile()
        :Inputs:
            *DBL*: :class:`pyOver.dataBook.DBTriqFM`
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
        """Read a ``triq`` annotated surface triangulation
        
        :Call:
            >>> DBF.ReadTriq(ftriq)
        :Inputs:
            *DBF*: :class:`pyOver.dataBook.DBTriqFM`
                Instance of TriqFM data book
            *ftriq*: :class:`str`
                Name of ``triq`` file
        :Versions:
            * 2017-03-29 ``@ddalle``: First version
        """
        # Check if the configuration file exists
        if os.path.isfile(self.conf):
            # Use that file (explicitly defined)
            fcfg = self.conf
        else:
            # Check for a mixsur file
            fmixsur = self.opts.get_DataBook_mixsur(self.comp)
            fusurp  = self.opts.get_DataBook_usurp(self.comp)
            # De-none
            if fmixsur is None: fmixsur = ''
            if fusurp  is None: fusurp = ''
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
        self.triq = cape.tri.Triq(ftriq, c=fcfg)
    
    # Preprocess triq file (convert from PLT)
    def PreprocessTriq(self, fq, **kw):
        """Perform any necessary preprocessing to create ``triq`` file
        
        :Call:
            >>> ftriq = DBF.PreprocessTriq(fq, qpbs=False, f=None)
        :Inputs:
            *DBL*: :class:`pyOver.dataBook.DBTriqFM`
                TriqFM data book
            *ftriq*: :class:`str`
                Name of q file
            *qpbs*: ``True`` | {``False``}
                Whether or not to create a script and submit it
            *f*: {``None``} | :class:`file`
                File handle if writing PBS script
        :Versions:
            * 2016-12-20 ``@ddalle``: First version
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
      
# class DBTriqFM

# Force/moment history
class CaseFM(cape.cfdx.dataBook.CaseFM):
    """
    This class contains methods for reading data about an the history of an
    individual component for a single case.  It reads the Tecplot file
    :file:`$proj_fm_$comp.dat` where *proj* is the lower-case root project name
    and *comp* is the name of the component.  From this file it determines
    which coefficients are recorded automatically.
    
    :Call:
        >>> FM = pyOver.dataBook.CaseFM(proj, comp)
    :Inputs:
        *proj*: :class:`str`
            Root name of the project
        *comp*: :class:`str`
            Name of component to process
    :Outputs:
        *FM*: :class:`pyOver.dataBook.FM`
            Instance of the force and moment class
        *FM.C*: :class:`list`\ [:class:`str`]
            List of coefficients
        *FM.i*: :class:`numpy.ndarray` shape=(0,)
            List of iteration numbers
        *FM.CA*: :class:`numpy.ndarray` shape=(0,)
            Axial force coefficient at each iteration
        *FM.CY*: :class:`numpy.ndarray` shape=(0,)
            Lateral force coefficient at each iteration
        *FM.CN*: :class:`numpy.ndarray` shape=(0,)
            Normal force coefficient at each iteration
        *FM.CLL*: :class:`numpy.ndarray` shape=(0,)
            Rolling moment coefficient at each iteration
        *FM.CLM*: :class:`numpy.ndarray` shape=(0,)
            Pitching moment coefficient at each iteration
        *FM.CLN*: :class:`numpy.ndarray` shape=(0,)
            Yaw moment coefficient at each iteration
    :Versions:
        * 2016-02-02 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, proj, comp):
        """Initialization method"""
        # Save component name
        self.comp = comp
        # Get the project rootname
        self.proj = proj
        # Expected name of the component history file
        ftmp = 'fomoco.tmp'
        fout = 'fomoco.out'
        frun = '%s.fomoco' % proj
        # Read stats from these files
        i_t, nc_t, ni_t = self.GetFomocoInfo(ftmp, comp)
        i_o, nc_o, ni_o = self.GetFomocoInfo(fout, comp)
        i_r, nc_r, ni_r = self.GetFomocoInfo(frun, comp)
        # Number of iterations
        ni = ni_t + ni_o + ni_r
        # Return empty if no data
        if ni == 0:
            self.MakeEmpty()
            return
        # Initialize data
        self.data = np.nan*np.ones((ni, 38))
        # Read the data.
        self.ReadFomocoData(frun, i_r, nc_r, ni_r, 0)
        self.ReadFomocoData(fout, i_o, nc_o, ni_o, ni_r)
        self.ReadFomocoData(ftmp, i_t, nc_t, ni_t, ni_r+ni_o)
        # Find non-NaN rows
        I = np.logical_not(np.isnan(self.data[:,0]))
        # Downselect
        self.data = self.data[I,:]
        # Save data as attributes
        self.SaveAttributes()
        
        
    # Get stats from a named FOMOCO file
    def GetFomocoInfo(self, fname, comp):
        """Get basic stats about an OVERFLOW fomoco file
        
        :Call:
            >>> ic, nc, ni = FM.GetFomocoInfo(fname, comp)
        :Inputs:
            *FM*: :class:`pyOver.dataBook.CaseFM`
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
            * 2016-02-03 ``@ddalle``: First version
        """
        # Check for the file
        if os.path.isfile(fname):
            # Get list of components
            comps = ReadFomocoComps(fname)
            # Number of components
            nc = len(comps)
            # Check if our component is present
            if comp in comps:
                # Index of the component.
                ic = comps.index(comp)
                # Number of (relevant) iterations
                ni = ReadFomocoNIter(fname, nc)
            else:
                # No useful iterations
                ic = 0
                # Number of (relevant) iterations
                ni = 0
            # Output
            return ic, nc, ni
        else:
            # No file
            return None, None, 0
            
    # Function to make empty one.
    def MakeEmpty(self, n=0):
        """Create empty *CaseFM* instance
        
        :Call:
            >>> FM.MakeEmpty()
        :Inputs:
            *FM*: :class:`pyOver.dataBook.CaseFM`
                Case force/moment history
        :Versions:
            * 2016-02-03 ``@ddalle``: First version
        """
        # Template entry
        A = np.nan * np.zeros(n)
        # Iterations
        self.i = A.copy()
        # Time
        self.t = A.copy()
        # Force coefficients
        self.CA = A.copy()
        self.CY = A.copy()
        self.CN = A.copy()
        # Moment coefficients
        self.CLL = A.copy()
        self.CLM = A.copy()
        self.CLN = A.copy()
        # Force pressure contributions
        self.CA_p = A.copy()
        self.CY_p = A.copy()
        self.CN_p = A.copy()
        # Force viscous contributions
        self.CA_v = A.copy()
        self.CY_v = A.copy()
        self.CN_v = A.copy()
        # Force momentum contributions
        self.CA_m = A.copy()
        self.CY_m = A.copy()
        self.CN_m = A.copy()
        # Moment pressure contributions
        self.CLL_p = A.copy()
        self.CLM_p = A.copy()
        self.CLN_p = A.copy()
        # Moment viscous contributions
        self.CLL_v = A.copy()
        self.CLM_v = A.copy()
        self.CLN_v = A.copy()
        # Moment momentum contributions
        self.CLL_m = A.copy()
        self.CLM_m = A.copy()
        self.CLN_m = A.copy()
        # Mass flow
        self.mdot = A.copy()
        # Areas
        self.A = A.copy()
        self.Ax = A.copy()
        self.Ay = A.copy()
        self.Az = A.copy()
        # Save a default list of columns and components.
        self.coeffs = [
            'CA',   'CY',   'CN',   'CLL',  'CLM',  'CLN',
            'CA_p', 'CY_p', 'CN_p', 'CA_v', 'CY_v', 'CN_v',
            'CA_m', 'CY_m', 'CN_m', 'CLL_p','CLM_p','CLN_p',
            'CLL_v','CLM_v','CLN_v','CLL_m','CLM_v','CLN_v',
            'mdot', 'A',    'Ax',   'Ay',   'Az'
        ]
        self.cols = ['i', 't'] + self.coeffs
        
    # Read data from a FOMOCO file
    def ReadFomocoData(self, fname, ic, nc, ni, n0=0):
        """Read data from a FOMOCO file with known indices and size
        
        :Call:
            >>> FM.ReadFomocoData(fname, ic, nc, ni, n0)
        :Inputs:
            *FM*: :class:`pyOver.dataBook.CaseFM`
                Force and moment history
            *fname*: :class:`str`
                Name of fomoco file
            *ic*: :class:`int`
                Index of *FM.comp* in list of components in *fname*
            *nc*: :class:`int`
                Number of components in *fname*
            *ni*: :class:`int`
                Number of iterations in *fname*
            *n0*: :class:`int`
                Number of iterations already read into *FM.data*
        :Versions:
            * 2016-02-03 ``@ddalle``: First version
        """
        # Exit if nothing to do
        if ni == 0: return
        # Check for file (in case any changes occurred before getting here)
        if not os.path.isfile(fname): return
        # Open the file
        f = open(fname)
        # Skip to start of first iteration
        f.seek(650*ic+81)
        # Number of iterations stored
        j = 0
        # Loop through iterations
        for i in range(ni):
            # Read data
            A = np.fromfile(f, sep=" ", count=38)
            # Check for iteration
            if len(A)==38 and (n0 == 0 or A[0] > self.data[n0-1,0]):
                # Save the data
                self.data[n0+j] = A
                # Increase count
                j += 1
            # Skip to next iteration
            f.seek(650*(nc-1) + 81 + f.tell())
        # Close the file
        f.close()
    
    # Function to make empty one.
    def SaveAttributes(self):
        """Save columns of *FM.data* as named attributes
        
        :Call:
            >>> FM.SaveAttributes()
        :Inputs:
            *FM*: :class:`pyOver.dataBook.CaseFM`
                Case force/moment history
        :Versions:
            * 2016-02-03 ``@ddalle``: First version
        """
        # Iterations
        self.i = self.data[:,0]
        # Time
        self.t = self.data[:,28]
        # Force pressure contributions
        self.CA_p = self.data[:,6]
        self.CY_p = self.data[:,7]
        self.CN_p = self.data[:,8]
        # Force viscous contributions
        self.CA_v = self.data[:,9]
        self.CY_v = self.data[:,10]
        self.CN_v = self.data[:,11]
        # Force momentum contributions
        self.CA_m = self.data[:,12]
        self.CY_m = self.data[:,13]
        self.CN_m = self.data[:,14]
        # Force coefficients
        self.CA = self.CA_p + self.CA_v + self.CA_m
        self.CY = self.CY_p + self.CY_v + self.CY_m
        self.CN = self.CN_p + self.CN_v + self.CN_m
        # Moment pressure contributions
        self.CLL_p = self.data[:,29]
        self.CLM_p = self.data[:,30]
        self.CLN_p = self.data[:,31]
        # Moment viscous contributions
        self.CLL_v = self.data[:,32]
        self.CLM_v = self.data[:,33]
        self.CLN_v = self.data[:,34]
        # Moment momentum contributions
        self.CLL_m = self.data[:,35]
        self.CLM_m = self.data[:,36]
        self.CLN_m = self.data[:,37]
        # Moment coefficients
        self.CLL = self.CLL_p + self.CLL_v + self.CLL_m
        self.CLM = self.CLM_p + self.CLM_v + self.CLM_m
        self.CLN = self.CLN_p + self.CLN_v + self.CLN_m
        # Mass flow
        self.mdot = self.data[:,27]
        # Areas
        self.A  = self.data[:,2]
        self.Ax = self.data[:,3]
        self.Ay = self.data[:,4]
        self.Az = self.data[:,5]
        # Save a default list of columns and components.
        self.coeffs = [
            'CA',   'CY',   'CN',   'CLL',  'CLM',  'CLN',
            'CA_p', 'CY_p', 'CN_p', 'CA_v', 'CY_v', 'CN_v',
            'CA_m', 'CY_m', 'CN_m', 'CLL_p','CLM_p','CLN_p',
            'CLL_v','CLM_v','CLN_v','CLL_m','CLM_v','CLN_v',
            'mdot', 'A',    'Ax',   'Ay',   'Az'
        ]
        self.cols = ['i', 't'] + self.coeffs
        
# class CaseFM


# Residual class
class CaseResid(cape.cfdx.dataBook.CaseResid):
    """OVERFLOW iterative residual history class
    
    This class provides an interface to residuals for a given case by reading
    the files ``resid.out``, ``resid.tmp``, ``run.resid``, ``turb.out``,
    ``species.out``, etc.
    
    :Call:
        >>> H = pyOver.dataBook.CaseResid(proj)
    :Inputs:
        *proj*: :class:`str`
            Project root name
    :Outputs:
        *H*: :class:`pyOver.databook.CaseResid`
            Instance of the residual histroy class
    :Versions:
        * 2016-02-03 ``@ddalle``: Started
    """
    # Initialization method
    def __init__(self, proj):
        """Initialization method
        
        :Versions:
            * 2016-02-03 ``@ddalle``: First version
        """
        # Save the prefix
        self.proj = proj
        # Initialize arrays.
        self.i = np.array([])
        self.L2   = np.array([])
        self.LInf = np.array([])
        
    # Representation method
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        # Display
        return "<pyOver.dataBook.CaseResid n=%i, prefix='%s'>" % (
            len(self.i), self.proj)
    # Copy the function
    __str__ = __repr__
        
    # Number of orders of magnitude of residual drop
    def GetNOrders(self, nStats=1):
        """Get the number of orders of magnitude of residual drop
        
        :Call:
            >>> nOrders = hist.GetNOrders(nStats=1)
        :Inputs:
            *hist*: :class:`pyCart.dataBook.CaseResid`
                Instance of the DataBook residual history
            *nStats*: :class:`int`
                Number of iterations to use for averaging the final residual
        :Outputs:
            *nOrders*: :class:`float`
                Number of orders of magnitude of residual drop
        :Versions:
            * 2015-01-01 ``@ddalle``: First versoin
        """
        # Process the number of usable iterations available.
        i = max(self.nIter-nStats, 0)
        # Get the maximum residual.
        L2Max = np.log10(np.max(self.L2))
        # Get the average terminal residual.
        L2End = np.log10(np.mean(self.L2[i:]))
        # Return the drop
        return L2Max - L2End
    
    # Read entire global residual history
    def ReadGlobalL2(self, grid=None):
        """Read entire global L2 history
        
        The file ``history.L2.dat`` is also updated.
        
        :Call:
            >>> H.ReadGlobalL2(grid=None)
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
            *grid*: {``None``} | :class:`int` | :class:`str`
                If used, read only one grid
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
            * 2017-04-19 ``@ddalle``: Added *grid* option
        """
        # Read the global history file
        if grid is None:
            self.i, self.L2 = self.ReadGlobalHist('history.L2.dat')
        # OVERFLOW file names
        frun = '%s.resid' % self.proj
        fout = 'resid.out'
        ftmp = 'resid.tmp'
        # Number of iterations read
        if len(self.i) > 0:
            # Last iteration
            n = int(self.i[-1])
        else:
            # Start from the beginning
            n = 0
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="L2", grid=grid)
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="L2", grid=grid)
        # Write the updated history (tmp file not safe to write here)
        if grid is None:
            self.WriteGlobalHist('history.L2.dat', self.i, self.L2)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="L2", grid=grid)
        
    # Read entire L-inf residual
    def ReadGlobalLInf(self, grid=None):
        """Read entire L-infinity norm history
        
        The file ``history.LInf.dat`` is also updated
        
        :Call:
            >>> H.ReadGlobalLInf(grid=None)
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
            *grid*: {``None``} | :class:`int` | :class:`str`
                If used, read only one grid
        :Versions:
            * 2016-02-06 ``@ddalle``: First version
            * 2017-04-19 ``@ddalle``: Added *grid* option
        """
        # Read the global history file
        if grid is None:
            self.i, self.LInf = self.ReadGlobalHist('histroy.LInf.dat')
        # OVERFLOW file names
        frun = '%s.resid' % self.proj
        fout = 'resid.out'
        ftmp = 'resid.tmp'
        # Number of iterations read
        if len(self.i) > 0:
            # Last iteration
            n = int(self.i[-1])
        else:
            # Start from the beginning
            n = 0
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="LInf", grid=grid)
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="LInf", grid=grid)
        # Write the updated history (tmp file not safe to write here)
        if grid is None:
            self.WriteGlobalHist('history.LInf.dat', self.i, self.L2)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="LInf", grid=grid)
        
    # Read turbulence L2 residual
    def ReadTurbResidL2(self, grid=None):
        """Read the entire L2 norm of the turbulence residuals
        
        The file ``history.turb.L2.dat`` is also updated
        
        :Call:
            >>> H.ReadTurbResidL2(grid=None)
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
            *grid*: {``None``} | :class:`int` | :class:`str`
                If used, read only one grid
        :Versions:
            * 2016-02-06 ``@ddalle``: First version
            * 2017-04-19 ``@ddalle``: Added *grid* option
        """
        # Read the global history file
        if grid is None:
            self.i, self.L2 = self.ReadGlobalHist('history.turb.L2.dat')
        # OVERFLOW file names
        frun = '%.turb' % self.proj
        fout = 'turb.out'
        ftmp = 'turb.tmp'
        # Number of iterations read
        if len(self.i) > 0:
            # Last iteration
            n = int(self.i[-1])
        else:
            # Start from the beginning
            n = 0
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="L2", grid=grid)
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="L2", grid=grid)
        # Write the updated history (tmp file not safe to write here)
        if grid is None:
            self.WriteGlobalHist('history.turb.L2.dat', self.i, self.L2)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="L2", grid=grid)
        
    # Read turbulence LInf residual
    def ReadTurbResidLInf(self, grid=None):
        """Read the global L-infinity norm of the turbulence residuals
        
        The file ``history.turb.LInf.dat`` is also updated
        
        :Call:
            >>> H.ReadTurbResidLInf()
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
            *grid*: {``None``} | :class:`int` | :class:`str`
                If used, read only one grid
        :Versions:
            * 2016-02-06 ``@ddalle``: First version
            * 2017-04-19 ``@ddalle``: Added *grid* option
        """
        # Read the global history file
        if grid is None:
            self.i, self.L2 = self.ReadglobalHist('history.turb.LInf.dat')
        # OVERFLOW file names
        frun = '%.turb' % self.proj
        fout = 'turb.out'
        ftmp = 'turb.tmp'
        # Number of iterations read
        if len(self.i) > 0:
            # Last iteration
            n = int(self.i[-1])
        else:
            # Start from the beginning
            n = 0
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="LInf", grid=grid)
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="LInf", grid=grid)
        # Write the updated history (tmp file not safe to write here)
        if grid is None:
            self.WriteGlobalHist('history.turb.LInf.dat', self.i, self.L2)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="LInf", grid=grid)
        
    # Read species L2 residual
    def ReadSpeciesResidL2(self, grid=None):
        """Read the global L2 norm of the species equations
        
        The file ``history.species.L2.dat`` is also updated
        
        :Call:
            >>> H.ReadSpeciesResidL2(grid=None)
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
            *grid*: {``None``} | :class:`int` | :class:`str`
                If used, read only one grid
        :Versions:
            * 2016-02-06 ``@ddalle``: First version
            * 2017-04-19 ``@ddalle``: Added *grid* option
        """
        # Read the global history file
        if grid is None:
            self.i, self.L2 = self.ReadglobalHist('history.species.L2.dat')
        # OVERFLOW file names
        frun = '%.species' % self.proj
        fout = 'species.out'
        ftmp = 'species.tmp'
        # Number of iterations read
        if len(self.i) > 0:
            # Last iteration
            n = int(self.i[-1])
        else:
            # Start from the beginning
            n = 0
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="L2", grid=grid)
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="L2", grid=grid)
        # Write the updated history (tmp file not safe to write here)
        if grid is None:
            self.WriteGlobalHist('history.species.L2.dat', self.i, self.L2)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="L2", grid=grid)
        
    # Read species LInf residual
    def ReadSpeciesResidLInf(self, grid=None):
        """Read the global L-infinity norm of the species equations
        
        The file ``history.species.LInf.dat`` is also updated
        
        :Call:
            >>> H.ReadSpeciesResidLInf(grid=None)
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
            *grid*: {``None``} | :class:`int` | :class:`str`
                If used, read only one grid
        :Versions:
            * 2016-02-06 ``@ddalle``: First version
            * 2017-04-19 ``@ddalle``: Added *grid* option
        """
        # Read the global history file
        if grid is None:
            self.i, self.L2 = self.ReadglobalHist('history.species.LInf.dat')
        # OVERFLOW file names
        frun = '%.species' % self.proj
        fout = 'species.out'
        ftmp = 'species.tmp'
        # Number of iterations read
        if len(self.i) > 0:
            # Last iteration
            n = int(self.i[-1])
        else:
            # Start from the beginning
            n = 0
        # Read the archival file
        self.ReadResidGlobal(frun, coeff="LInf", grid=grid)
        # Read the intermediate file
        self.ReadResidGlobal(fout, coeff="LInf", grid=grid)
        # Write the updated history (tmp file not safe to write here)
        if grid is None:
            self.WriteGlobalHist('history.species.LInf.dat', self.i, self.L2)
        # Read the temporary file
        self.ReadResidGlobal(ftmp, coeff="LInf", grid=grid)
    
    # Read a consolidated history file
    def ReadGlobalHist(self, fname):
        """Read a condensed global residual file for faster read times
        
        :Call:
            >>> i, L = H.ReadGlobalHist(fname)
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
            *i*: :class:`numpy.ndarray` (:class:
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        # Check for file.
        if not os.path.isfile(fname):
            # Return empty arrays
            return np.array([]), np.array([])
        # Try to read the file
        try:
            # Read the file.
            A = np.loadtxt(fname)
            # Split into columns
            return A[:,0], A[:,1]
        except Exception:
            # Reading file failed
            return np.array([]), np.array([])
    
    # Write a consolidated history file
    def WriteGlobalHist(self, fname, i, L, n=None):
        """Write a condensed global residual file for faster read times
        
        :Call:
            >>> H.WriteGlobalHist(fname, i, L, n=None)
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
            *i*: :class:`np.ndarray` (:class:`float` | :class:`int`)
                Vector of iteration numbers
            *L*: :class:`np.ndarray`\ [:class:`float`]
                Vector of residuals to write
            *n*: :class:`int` | ``None``
                Last iteration already written to file.
        :Versions:
            * 2016-02-04 ``@ddalle``: First version
        """
        # Default number of lines to skip
        if n is None:
            # Query the file.
            if os.path.isfile(fname):
                try:
                    # Read the last line of the file
                    line = bin.tail(fname)
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
        if len(I) == 0: return
        # Index to start at
        istart = I[0]
        # Append to the file
        f = open(fname, 'a')
        # Loop through the lines
        for j in range(istart, len(i)):
            # Write iteration
            f.write('%8i %14.7E\n' % (i[j], L[j]))
        # Close the file.
        f.close()

    # Read a global residual file
    def ReadResidGlobal(self, fname, coeff="L2", n=None, grid=None):
        """Read a global residual using :func:`numpy.loadtxt` from one file
        
        :Call:
            >>> i, L2 = H.ReadResidGlobal(fname, coeff="L2", **kw)
            >>> i, LInf = H.ReadResidGlobal(fname, coeff="LInf", **kw)
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
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
            * 2016-02-04 ``@ddalle``: First version
            * 2017-04-19 ``@ddalle``: Added *grid* option
        """
        # Check for individual grid
        if grid is not None:
            # Pass to individual grid reader
            iL = self.ReadResidGrid(fname, grid=grid, coeff=coeff, n=n)
            # Quit.
            return iL
        # Check for the file
        if not os.path.isfile(fname): return
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
            if len(self.i) == 0:
                # No iterations
                n = 0
            else:
                # Use last current iter
                n = int(max(self.i))
        # Number of iterations to skip
        nIterSkip = max(0, n-i0+1)
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
            cols = (1,3)
            nc = 2
            # Coefficient
            c = 'LInf'
        else:
            # Read the iter, L2 norm, nPts
            cols = (1,2,13)
            nc = 3
            # Field name
            c = 'L2'
        # Read the file
        A = np.loadtxt(fname, skiprows=nSkip, usecols=cols)
        # Reshape the data
        B = np.reshape(A[:nIterRead*nGrid,:], (nIterRead, nGrid, nc))
        # Get iterations
        i = B[:,0,0]
        # Filter iterations greater than *n*
        I = i > n
        i = i[I]
        # Exit if no iterations
        if len(i) == 0: return
        # Get global residuals
        if c == "L2":
            # Get weighted sum
            L = np.sum(B[I,:,1]*B[I,:,2]**2, axis=1)
            # Total grid points in each iteration
            N = np.sum(B[I,:,2], axis=1)
            # Divide by number of grid points, and take square root
            L = np.sqrt(L/N)
            # Append to data
            self.L2 = np.hstack((self.L2, L))
        else:
            # Get the maximum value
            L = np.max(B[I,:,1], axis=1)
            # Append to data
            self.LInf = np.hstack((self.LInf, L))
        # Check for issues
        if np.any(np.diff(i) < 0):
            # Warning
            print("  Warning: file '%s' contains non-ascending iterations" %
                fname)
        # Append to data
        self.i = np.hstack((self.i, i))
        # Output
        return i, L

    # Read a global residual file
    def ReadResidGrid(self, fname, grid=None, coeff="L2", n=None):
        """Read a global residual using :func:`numpy.loadtxt` from one file
        
        :Call:
            >>> i, L2 = H.ReadResidGrid(fname, grid=None, coeff="L2", **kw)
            >>> i, LInf = H.ReadResidGrid(fname, grid=None, coeff="LInf", **kw)
        :Inputs:
            *H*: :class:`pyOver.dataBook.CaseResid`
                Iterative residual history class
            *fname*: :class:`str`
                Name of file to process
            *grid*: {``None``} | :class:`int` | :class:`str`
                If used, read history of a single grid
            *coeff*: :class:`str`
                Name of coefficient to read
            *n*: :class:`int` | ``None``
                Number of last iteration that's already processed
        :Outputs:
            *i*: :class:`np.ndarray`\ [:class:`float`]
                Array of iteration numbers
            *L2*: :class:`np.ndarray`\ [:class:`float`]
                Array of weighted global L2 norms
            *LInf*: :class:`np.ndarray`\ [:class:`float`]
                Array of global L-infinity norms
        :Versions:
            * 2017-04-19 ``@ddalle``: First version
        """
        # Check for the file
        if not os.path.isfile(fname): return None, None
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
            if len(self.i) == 0:
                # No iterations
                n = 0
            else:
                # Use last current iter
                n = int(max(self.i))
        # Individual grid type
        tg = type(grid).__name__
        # Individual grid
        if grid is None:
            # Read all grids
            kGrid = nGrid
        elif tg.startswith('unicode') or tg.startswith('str'):
            # Figure out grid number
            grids = ReadResidGrids(fname)
            # Check presence
            if grid not in grids:
                raise ValueError("Could not find grid '%s'" % grid)
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
            cols = [1,3]
            nc = 2
            # Coefficient
            c = 'LInf'
        else:
            # Read the iter, L2 norm, nPts
            cols = [1,2,13]
            nc = 3
            # Field name
            c = 'L2'
        # Initialize matrix
        B = np.zeros((nIterRead, kGrid, nc))
        # Open the file
        f = open(fname, 'r')
        # Skip desired number of rows
        f.seek(nSkip*218)
        # Check if we should skip to grid *grid*
        if grid is not None:
            f.seek(iGrid*218, 1)
        # Loop through iterations
        for j in np.arange(nIterRead):
            # Loop through grids
            for k in KGrid:
                # Read data
                bjk = np.fromfile(f, sep=" ", count=-1)
                # Save it
                B[j,k,:] = bjk[cols]
                # Skip over the string
                f.seek(26, 1)
            # Skip rows if appropriate
            if grid is not None:
                f.seek((nGrid-1)*218, 1)
        # Close the file
        f.close()
        # Get iterations
        i = B[:,0,0]
        # Filter iterations greater than *n*
        I = i > n
        i = i[I]
        # Exit if no iterations
        if len(i) == 0: return
        # Get global residuals
        if c == "L2":
            # Get weighted sum
            L = np.sum(B[I,:,1]*B[I,:,2]**2, axis=1)
            # Total grid points in each iteration
            N = np.sum(B[I,:,2], axis=1)
            # Divide by number of grid points, and take square root
            L = np.sqrt(L/N)
            # Append to data
            self.L2 = np.hstack((self.L2, L))
        else:
            # Get the maximum value
            L = np.max(B[I,:,1], axis=1)
            # Append to data
            self.LInf = np.hstack((self.LInf, L))
        # Check for issues
        if np.any(np.diff(i) < 0):
            # Warning
            print("  Warning: file '%s' contains non-ascending iterations" %
                fname)
        # Append to data
        self.i = np.hstack((self.i, i))
        # Output
        return i, L
    
    # Plot L2 norm
    def PlotL2(self, n=None, nFirst=None, nLast=None, **kw):
        """Plot the L2 residual
        
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
            * 2014-11-12 ``@ddalle``: First version
            * 2014-12-09 ``@ddalle``: Moved to :class:`AeroPlot`
            * 2015-02-15 ``@ddalle``: Transferred to :class:`dataBook.Aero`
            * 2015-03-04 ``@ddalle``: Added *nStart* and *nLast*
            * 2015-10-21 ``@ddalle``: Referred to :func:`PlotResid`
        """
        # Y-label option
        ylbl = kw.get('YLabel', 'L2 Residual')
        # Plot 'L2Resid'
        return self.PlotResid('L2', 
            n=n, nFirst=nFirst, nLast=nLast, YLabel='L2 Residual')
# class CaseResid

