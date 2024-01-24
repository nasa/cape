r"""
:mod:`cape.pyfun.dataBook`: Post-processing for FUN3D data
=============================================================

This module contains functions for reading and processing forces,
moments, and other statistics from cases in a trajectory.  Data books
are usually created by using the
:func:`cape.pyfun.cntl.Cntl.ReadDataBook` function.

    .. code-block:: python

        # Read FUN3D control instance
        cntl = pyFun.Cntl("pyFun.json")
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
:class:`cape.pyfun.runmatrix.RunMatrix`, so it is a more involved process.

Data book modules are also invoked during update and reporting
command-line calls.

    .. code-block:: console

        $ pyfun --aero
        $ pyfun --ll
        $ pyfun --pt
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
    * :mod:`cape.pyfun.lineLoad`
    * :mod:`cape.options.DataBook`
    * :mod:`cape.pyfun.options.DataBook`
"""

# Standard library modules
import os
import re
import glob

# Third-party modules
import numpy as np

# Local imports
from . import case
from . import lineLoad
from . import pointSensor
from . import plt
from ..cfdx import dataBook
from ..attdb.ftypes import tsvfile


# Radian -> degree conversion
deg = np.pi / 180.0

# Column names for FM files
COLNAMES_FM = {
    "Iteration": dataBook.CASE_COL_ITRAW,
    "C_L": "CL",
    "C_D": "CD",
    "C_M_x": "CLL",
    "C_M_y": "CLM",
    "C_M_z": "CLN",
    "C_x": "CA",
    "C_y": "CY",
    "C_z": "CN",
    "C_Lp": "CLp",
    "C_Dp": "CDp",
    "C_Lv": "CLv",
    "C_Dv": "CDv",
    "C_M_xp": "CLLp",
    "C_M_yp": "CLMp",
    "C_M_zp": "CLNp",
    "C_M_xv": "CLLv",
    "C_M_yv": "CLMv",
    "C_M_zv": "CLNv",
    "C_xp": "CAp",
    "C_yp": "CYp",
    "C_zp": "CNp",
    "C_xv": "CAv",
    "C_yv": "CYv",
    "C_zv": "CNv",
    "Mass flow": "mdot",
    "<greek>r</greek>": "rho",
    "p/p<sub>0</xub>": "phat",
    "p<sub>t</sub>/p<sub>0</sub>": "p0hat",
    "T<sub>t</sub>": "T0",
    "T<sub>RMS</xub>": "Trms",
    "Mach": "mach",
    "Simulation Time": dataBook.CASE_COL_TRAW,
}

# Column names for primary history, {PROJ}_hist.dat
COLNAMES_HIST = {
    "Iteration": dataBook.CASE_COL_ITRAW,
    "C_L": "CL",
    "C_D": "CD",
    "C_M_x": "CLL",
    "C_M_y": "CLM",
    "C_M_z": "CLN",
    "C_x": "CA",
    "C_y": "CY",
    "C_z": "CN",
    "C_Lp": "CLp",
    "C_Dp": "CDp",
    "C_Lv": "CLv",
    "C_Dv": "CDv",
    "C_M_xp": "CLLp",
    "C_M_yp": "CLMp",
    "C_M_zp": "CLNp",
    "C_M_xv": "CLLv",
    "C_M_yv": "CLMv",
    "C_M_zv": "CLNv",
    "C_xp": "CAp",
    "C_yp": "CYp",
    "C_zp": "CNp",
    "C_xv": "CAv",
    "C_yv": "CYv",
    "C_zv": "CNv",
    "Wall Time": "WallTime",
    "Simulation_Time": dataBook.CASE_COL_TRAW,
}

# Column names for fractional time step history, {PROJ}_subhist.dat
COLNAMES_SUBHIST = {
    "Fractional_Time_Step": dataBook.CASE_COL_ITRAW + "_sub",
    "R_1": "R_1_sub",
    "R_2": "R_2_sub",
    "R_3": "R_3_sub",
    "R_4": "R_4_sub",
    "R_5": "R_5_sub",
    "R_6": "R_6_sub",
    "C_x": "CA_sub",
    "C_y": "CY_sub",
    "C_z": "CN_sub",
    "C_M_x": "CLL_sub",
    "C_M_y": "CLM_sub",
    "C_M_z": "CLN_sub",
    "C_L": "CL_sub",
    "C_D": "CD_sub",
}


# Aerodynamic history class
class DataBook(dataBook.DataBook):
    r"""This class provides an interface to the data book for a given
    CFD run matrix.

    :Call:
        >>> DB = pyFun.dataBook.DataBook(x, opts)
    :Inputs:
        *x*: :class:`cape.pyfun.runmatrix.RunMatrix`
            The current pyFun trajectory (i.e. run matrix)
        *opts*: :class:`cape.pyfun.options.Options`
            Global pyFun options instance
    :Outputs:
        *DB*: :class:`cape.pyfun.dataBook.DataBook`
            Instance of the pyFun data book class
    """
  # ===========
  # Readers
  # ===========
    # Initialize a DBComp object
    def ReadDBComp(self, comp, check=False, lock=False):
        r"""Initialize data book for one component

        :Call:
            >>> DB.ReadDBComp(comp, check=False, lock=False)
        :Inputs:
            *DB*: :class:`cape.pyfun.dataBook.DataBook`
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
        # Read the data book
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
            *DB*: :class:`cape.pyfun.dataBook.DataBook`
                Instance of pyFun data book class
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
            # Confirm lock if necessary.
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

    # Read TriqPoint components
    def ReadTriqPoint(self, comp, check=False, lock=False, **kw):
        r"""Read a TriqPoint data book if not already present

        :Call:
            >>> DB.ReadTriqPoint(comp, check=False, lock=False, **kw)
        :Inputs:
            *DB*: :class:`cape.pyfun.dataBook.DataBook`
                Instance of pyFun data book class
            *comp*: :class:`str`
                Name of TriqFM component
            *check*: ``True`` | {``False``}
                Whether or not to check LOCK status
            *lock*: ``True`` | {``False``}
                If ``True``, wait if the LOCK file exists
            *pts*: {``None``} | :class:`list`\ [:class:`str`]
                List of points to read (default is read from *DB.opts*)
            *pt*: {``None``} | :class:`str`
                Individual point to read
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
            * 2017-10-11 ``@ddalle``: From :func:`ReadTriqFM`
        """
        # Initialize if necessary
        try:
            self.TriqPoint
        except Exception:
            self.TriqPoint = {}
        # Get point list
        pts = kw.get("pts", kw.get("pt"))
        # Check type
        if pts is None:
            # Default list
            pts = self.opts.get_DataBookPoints(comp)
        elif type(pts).__name__ not in ["list", "ndarray"]:
            # One point; convert to list
            pts = [pts]
        # Try to access the TriqPoint database
        try:
            # Check if present
            DBPG = self.TriqPoint[comp]
            # Loop through points to check if they're present
            for pt in pts:
                # Check if present
                if pt in DBPG:
                    continue
                # Otherwise/read it
                DBPG.ReadPointSensor(pt)
                # Add to the list
                DBPG.pts.append(pt)
            # Confirm lock if necessary.
            if lock:
                self.TriqPoint[comp].Lock()
        except Exception:
            # Safely go to root directory
            fpwd = os.getcwd()
            os.chdir(self.RootDir)
            # Read data book
            self.TriqPoint[comp] = pointSensor.DBTriqPointGroup(
                self.x, self.opts, comp, pts=pts,
                RootDir=self.RootDir, check=check, lock=lock)
            # Return to starting position
            os.chdir(fpwd)

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
            *DB*: :class:`cape.cfdx.dataBook.DataBook`
                Instance of data book class
        :Outputs:
            *H*: :class:`cape.pyfun.dataBook.CaseResid`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseResid(self.proj)

    # Read case FM history
    def ReadCaseFM(self, comp):
        r"""Read a :class:`CaseFM` object

        :Call:
            >>> FM = DB.ReadCaseFM(comp)
        :Inputs:
            *DB*: :class:`cape.cfdx.dataBook.DataBook`
                Instance of data book class
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *FM*: :class:`cape.pyfun.dataBook.CaseFM`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseFM(self.proj, comp)


# Component data book
class DBComp(dataBook.DBComp):
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
        *DBF*: :class:`cape.pyfun.dataBook.DBTriqFM`
            Instance of TriqFM data book
    :Versions:
        * 2017-03-28 ``@ddalle``: v1.0
    """

    # Get file
    def GetTriqFile(self):
        r"""Get most recent ``triq`` file and its associated iterations

        :Call:
            >>> qtriq, ftriq, n, i0, i1 = DBF.GetTriqFile()
        :Inputs:
            *DBF*: :class:`cape.pyfun.dataBook.DBTriqFM`
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
        fplt, n, i0, i1 = case.GetPltFile()
        if fplt is None:
            return False, None, None, None, None
        # Check for iteration resets
        nh, ns = case.GetHistoryIter()
        # Add in the last iteration number before restart
        if nh is not None:
            i0 += nh
            i1 += nh
        # Get the corresponding .triq file name
        ftriq = fplt.rstrip('.plt') + '.triq'
        # Check if the TRIQ file exists
        if os.path.isfile(ftriq):
            # No conversion needed
            qtriq = False
        else:
            # Need to convert PLT file to TRIQ
            qtriq = True
        # Output
        return qtriq, ftriq, n, i0, i1

    # Preprocess triq file (convert from PLT)
    def PreprocessTriq(self, ftriq, **kw):
        r"""Perform any necessary preprocessing to create ``triq`` file

        :Call:
            >>> DBL.PreprocessTriq(ftriq, i=None)
        :Inputs:
            *DBF*: :class:`cape.pyfun.dataBook.DBTriqFM`
                Instance of TriqFM data book
            *ftriq*: :class:`str`
                Name of triq file
            *i*: {``None``} | :class:`int`
                Case index (else read from :file:`conditions.json`)
        :Versions:
            * 2017-03-28 ``@ddalle``: v1.0
        """
        # Get name of plt file
        fplt = ftriq.rstrip('triq') + 'plt'
        # Get case index
        i = kw.get('i')
        # Read Mach number
        if i is None:
            # Read from :file:`conditions.json`
            mach = case.ReadConditions('mach')
        else:
            # Get from trajectory
            mach = self.x.GetMach(i)
        # Output format
        fmt = self.opts.get_DataBookTriqFormat(self.comp)
        # Read the plt information
        plt.Plt2Triq(fplt, ftriq, mach=mach, fmt=fmt)


# Force/moment history
class CaseFM(dataBook.CaseFM):
    r"""Iterative force & moment histories for one case, one component

    This class contains methods for reading data about an the history
    of an individual component for a single case.  It reads the Tecplot
    file :file:`$proj_fm_$comp.dat` where *proj* is the lower-case root
    project name and *comp* is the name of the component.  From this
    file it determines which coefficients are recorded automatically.

    :Call:
        >>> fm = CaseFM(proj, comp)
    :Inputs:
        *proj*: :class:`str`
            Root name of the project
        *comp*: :class:`str`
            Name of component to process
    :Outputs:
        *fm*: :class:`CaseFM`
            Instance of the force and moment class
    :Versions:
        * 2014-11-12 ``@ddalle``: v0.1; starter version
        * 2015-10-16 ``@ddalle``: v1.0
        * 2016-05-05 ``@ddalle``: v1.1; handle adaptive cases
        * 2016-10-28 ``@ddalle``: v1.2; catch iteration resets
    """
    # Initialization method
    def __init__(self, proj: str, comp: str, **kw):
        r"""Initialization method

        :Versions:
            * 2025-10-16 ``@ddalle``: v1.0
            * 2024-01-23 ``@ddalle``: v2.0; DataKit
        """
        # Get the project rootname
        self.proj = proj
        # Use parent initializer
        dataBook.CaseFM.__init__(self, comp, **kw)

    # Get working folder for flow
    def get_flow_folder(self) -> str:
        r"""Get the working folder for primal solutions

        This will be either ``""`` (base dir) or ``"Flow"``

        :Call:
            >>> workdir = fm.get_flow_folder()
        :Inputs:
            *fm*: :class:`CaseFM`
                Force & moment iterative history
        :Outputs:
            *workdir*: ``""`` | ``"Flow"``
                Current working folder for primal (flow) solutions
        :Versions:
            * 2024-01-23 ``@ddalle``: v1.0
        """
        # Check for ``Flow/`` folder
        return "Flow" if os.path.isdir("Flow") else ""

    # Get list of files to read
    def get_filelist(self) -> list:
        r"""Get list of files to read

        :Call:
            >>> filelist = fm.get_filelist()
        :Inputs:
            *fm*: :class:`CaseFM`
                Component iterative history instance
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to read to construct iterative history
        :Versions:
            * 2024-01-23 ``@ddalle``: v1.0; from old __init__()
        """
        # Check for ``Flow`` folder
        workdir = self.get_flow_folder()
        # Quick function to join workdir and file name
        fw = lambda fname: os.path.join(workdir, fname)
        # Get project and component name
        proj = self.proj
        comp = self.comp
        compl = comp.lower()
        # Expected name of the component history file(s)
        fname = fw(f"{proj}_fm_{comp}.dat")
        fnamel = fw(f"{proj}_fm_{compl}.dat")
        # Patters for multiple-file scenarios
        fglob1 = fw(f"{proj}_fm_{comp}.[0-9][0-9].dat")
        fglob2 = fw(f"{proj}[0-9][0-9]_fm_{comp}.dat")
        fglob3 = fw(f"{proj}[0-9][0-9]_fm_{comp}.[0-9][0-9].dat")
        # Lower-case versions
        fglob1l = fw(f"{proj}_fm_{compl}.[0-9][0-9].dat")
        fglob2l = fw(f"{proj}[0-9][0-9]_fm_{compl}.dat")
        fglob3l = fw(f"{proj}[0-9][0-9]_fm_{compl}.[0-9][0-9].dat")
        # List of files
        filelist = []
        # Check which scenario we're in
        if os.path.isfile(fname):
            # Single project + original case; check for history resets
            glob1 = glob.glob(fglob1)
            glob1.sort()
            # Add in main file name
            filelist = glob1 + [fname]
        elif os.path.isfile(fnamel):
            # Single project + original case; check for history resets
            glob1 = glob.glob(fglob1l)
            glob1.sort()
            # Add in main file name
            filelist = glob1 + [fnamel]
        else:
            # Multiple projects; try original case first
            glob2 = glob.glob(fglob2)
            glob3 = glob.glob(fglob3)
            # Check for at least one match
            if len(glob2 + glob3) > 0:
                # Save original case
                filelist = glob2 + glob3
            else:
                # Find lower-case matches
                glob2 = glob.glob(fglob2l)
                glob3 = glob.glob(fglob3l)
                # Save lower-case versions
                filelist = glob2 + glob3
        # Sort whatever list we've god
        filelist.sort()
        # Output
        return filelist

    # Read a data file
    def readfile(self, fname: str) -> dict:
        r"""Read a Tecplot iterative history file

        :Call:
            >>> db = fm.readfile(fname)
        :Inputs:
            *fm*: :class:`CaseFM`
                Single-component iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *db*: :class:`tsvfile.TSVTecDatFile`
                Data read from *fname*
        :Versions:
            * 2024-01-23 ``@ddalle``: v1.0
        """
        # Read the Tecplot file
        db = tsvfile.TSVTecDatFile(fname, Translators=COLNAMES_FM)
        # Modify iteration & time histories
        self._fix_iter(db)
        # Output
        return db

    # Function to fix iteration histories of one file
    def _fix_iter(self, db: tsvfile.TSVTecDatFile):
        r"""Fix iteration and time histories for FUN3D resets

        :Call:
            >>> _fix_iter(h, db)
        :Versions:
            * 2024-01-23 ``@ddalle``: v1.0
        """
        # Get iterations and time
        i_solver = db.get(dataBook.CASE_COL_ITRAW)
        t_solver = db.get(dataBook.CASE_COL_TRAW)
        # Get current last iter
        i_last = self.get_lastiter()
        # Copy to actual
        i_cape = i_solver.copy()
        # Required delta for iteration counter
        di = max(0, i_last - i_solver[0] + 1)
        # Modify history
        i_cape += di
        # Save iterations
        db.save_col(dataBook.CASE_COL_ITERS, i_cape)
        # Modify time history
        if (t_solver is None) or (t_solver[0] < 0):
            # No time histories
            t_raw = np.full(i_solver.size, np.nan)
            t_cape = np.full(i_solver.shape, np.nan)
            # Save placeholders for raw time
            db.save_col(dataBook.CASE_COL_TRAW, t_raw)
        else:
            # Get last time value
            t_last = self.get_maxtime()
            # Copy to actual
            t_cape = t_solver.copy()
            # Required delta for times to be ascending
            dt = max(0.0, np.floor(t_last - 2*t_solver[0] + t_solver[1]))
            # Modify time histories
            t_cape += dt
        # Save time histories
        db.save_col(dataBook.CASE_COL_TIME, t_cape)
        # Output
        return db


# Class to keep track of residuals
class CaseResid(dataBook.CaseResid):
    r"""FUN3D iterative history class

    This class provides an interface to residuals, CPU time, and
    similar data for a given case

    :Call:
        >>> hist = CaseResid(proj)
    :Inputs:
        *proj*: :class:`str`
            Project root name
    :Outputs:
        *hist*: :class:`cape.pyfun.dataBook.CaseResid`
            Instance of the run history class
    """
    # Base columns
    _base_cols = (
        'i',
        'R_1',
        'R_2',
        'R_3',
        'R_4',
        'R_5',
        'R_6',
        'R_7',
        'L2Resid',
        'L2Resid0'
    )
    # Columns other than *i*
    _base_coeffs = (
        'R_1',
        'R_2',
        'R_3',
        'R_4',
        'R_5',
        'R_6',
        'R_7',
        'L2Resid',
        'L2Resid0'
    )

    # Initialization method
    def __init__(self, proj: str, **kw):
        r"""Initialization method

        :Versions:
            * 2015-10-21 ``@ddalle``: v1.0
            * 2016-10-28 ``@ddalle``: v1.1; catch iteration resets
            * 2023-01-10 ``@ddalle``: v2.0; subclass to ``DataKit``
        """
        # Save the project root name
        self.proj = proj
        # Pass to parent class
        dataBook.CaseResid.__init__(self, **kw)

    # Get list of files to read
    def get_filelist(self) -> list:
        r"""Get list of files to read

        :Call:
            >>> filelist = h.get_filelist()
        :Inputs:
            *fm*: :class:`CaseResid`
                Component iterative history instance
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to read to construct iterative history
        :Versions:
            * 2024-01-23 ``@ddalle``: v1.0; from old __init__()
        """
        # Check for ``Flow`` folder
        workdir = self.get_flow_folder()
        # Quick function to join workdir and file name
        fw = lambda fname: os.path.join(workdir, fname)
        # Get project and component name
        proj = self.proj
        # Expected name of the component history file(s)
        fname = fw(f"{proj}_hist.dat")
        # Patters for multiple-file scenarios
        fglob1 = fw(f"{proj}_hist.[0-9][0-9].dat")
        fglob2 = fw(f"{proj}[0-9][0-9]_hist.dat")
        fglob3 = fw(f"{proj}[0-9][0-9]_hist.[0-9][0-9].dat")
        # List of files
        filelist = []
        # Check which scenario we're in
        if os.path.isfile(fname):
            # Single project + original case; check for history resets
            glob1 = glob.glob(fglob1)
            glob1.sort()
            # Add in main file name
            filelist = glob1 + [fname]
        else:
            # Multiple projects; try original case first
            glob2 = glob.glob(fglob2)
            glob3 = glob.glob(fglob3)
            # Combine both matches
            filelist = glob2 + glob3
        # Sort whatever list we've god
        filelist.sort()
        # Output
        return filelist

    # Get working folder for flow
    def get_flow_folder(self) -> str:
        r"""Get the working folder for primal solutions

        This will be either ``""`` (base dir) or ``"Flow"``

        :Call:
            >>> workdir = fm.get_flow_folder()
        :Inputs:
            *fm*: :class:`CaseFM`
                Force & moment iterative history
        :Outputs:
            *workdir*: ``""`` | ``"Flow"``
                Current working folder for primal (flow) solutions
        :Versions:
            * 2024-01-23 ``@ddalle``: v1.0
        """
        # Check for ``Flow/`` folder
        return "Flow" if os.path.isdir("Flow") else ""

    # Read a data file
    def readfile(self, fname: str) -> dict:
        r"""Read a Tecplot iterative history file

        :Call:
            >>> db = fm.readfile(fname)
        :Inputs:
            *fm*: :class:`CaseFM`
                Single-component iterative history instance
            *fname*: :class:`str`
                Name of file to read
        :Outputs:
            *db*: :class:`tsvfile.TSVTecDatFile`
                Data read from *fname*
        :Versions:
            * 2024-01-23 ``@ddalle``: v1.0
        """
        # Read the Tecplot file
        db = tsvfile.TSVTecDatFile(fname, Translators=COLNAMES_HIST)
        # Fix iterative histories
        di = self._fix_iter(db)
        # Read subiterations, if possible
        dbsub = self.read_subhist(fname, di)
        # Merge
        for col in dbsub:
            db.save_col(col, dbsub[col])
        # Calculate L2 norms by adding up R_{n} contributions
        self._build_l2(db)
        # Output
        return db

    # Calculate L2 norm(s)
    def _build_l2(self, db: tsvfile.TSVTecDatFile):
        r"""Calculate *L2* norm for initial, final, and subiter

        :Versions:
            * 2024-01-24 ``@ddalle``: v1.0
        """
        # Loop through three suffixes
        for suf in ("", "_0", "_sub"):
            # Column name for iteration
            icol = dataBook.CASE_COL_ITERS + suf
            # Column name for initial residual/value
            rcol = f"L2Resid{suf}"
            # Get iterations corresponding to this sufix
            iters = self.get(icol)
            # Skip if not present
            if iters is None:
                continue
            # Initialize cumulative residual squared
            L2squared = np.zeros_like(iters, dtype="float")
            # Loop through potential residuals
            for c in ("R_1", "R_2", "R_3", "R_4", "R_5", "R_6"):
                # Check for baseline
                col = c + suf
                # Get values
                v = db.get(col)
                # Assemble
                if v is not None:
                    L2squared += v*v
            # Save residuals
            db.save_col(rcol, np.sqrt(L2squared))

    # Read subhistory files
    def read_subhist(self, fname: str, di: float) -> dict:
        r"""Read a Tecplot sub-iterative history file

        These files, e.g. ``{PROJECT}_subhist.dat``, are written when
        the solver is in time-accurate mode.

        :Call:
            >>> db = fm.read_subhist(fname, di)
        :Inputs:
            *fm*: :class:`CaseFM`
                Single-component iterative history instance
            *fname*: :class:`str`
                Name of main iterative history file
            *di*: :class:`float` | :class:`int`
                Iteration shift to correct for FUN3D counter restarts
        :Outputs:
            *db*: :class:`tsvfile.TSVTecDatFile`
                Data read from *fname*
        :Versions:
            * 2024-01-24 ``@ddalle``: v1.0
        """
        # Patterns for subhist files
        pat1 = fname.replace("hist", "subhist")
        pat2 = fname.replace("hist", "subhist.old[0-9][0-9]")
        # Find matches
        subhist_files = sorted(glob.glob(pat2)) + glob.glob(pat1)
        # Initialize in case of no matches
        db = {}
        # Loop through files
        for j, subhist_file in enumerate(subhist_files):
            # Read it
            dbj = self.readfile_subhist(subhist_file, di)
            # Initialize
            if j == 0:
                # Initial
                db = dbj
            else:
                # Combine
                for col in dbj:
                    db[col] = np.hstack((db[col], dbj[col]))
        # Output
        return db

    # Read subhistory file
    def readfile_subhist(self, fname: str, di: float) -> dict:
        r"""Read a Tecplot sub-iterative history file

        These files, e.g. ``{PROJECT}_subhist.dat``, are written when
        the solver is in time-accurate mode.

        :Call:
            >>> db = fm.readfile_subhist(fname, di)
        :Inputs:
            *fm*: :class:`CaseFM`
                Single-component iterative history instance
            *fname*: :class:`str`
                Name of file to read
            *di*: :class:`float` | :class:`int`
                Iteration shift to correct for FUN3D counter restarts
        :Outputs:
            *db*: :class:`tsvfile.TSVTecDatFile`
                Data read from *fname*
        :Versions:
            * 2024-01-23 ``@ddalle``: v1.0
        """
        # Read the _subhist.dat file
        db = tsvfile.TSVTecDatFile(fname, Translators=COLNAMES_SUBHIST)
        # Get the raw subiteration reported by FUN3D
        i_raw = db[dataBook.CASE_COL_ITRAW + "_sub"]
        # Modify iteration to global history value
        i_cape = i_raw + di
        # Save that
        db.save_col(dataBook.CASE_COL_ITERS + "_sub", i_cape)
        # Find indices of first subiteration at each major iteration
        mask0 = (i_raw == np.floor(i_raw))
        # Loop through cols
        for col in db.cols:
            # Create new column marking beginning of major iter
            col0 = col.replace("_sub", "_0")
            # Save
            db.save_col(col0, db[col][mask0])
        # Output
        return db

    # Function to fix iteration histories of one file
    def _fix_iter(self, db: tsvfile.TSVTecDatFile):
        r"""Fix iteration and time histories for FUN3D resets

        :Call:
            >>> _fix_iter(h, db)
        :Versions:
            * 2024-01-23 ``@ddalle``: v1.0
        """
        # Get last time step
        # (to check if we're switching time-accurate <-> steady-state)
        t_last = self.get_lasttime()
        # Get iterations and time
        i_solver = db.get(dataBook.CASE_COL_ITRAW)
        t_solver = db.get(dataBook.CASE_COL_TRAW)
        # Check if last reported iter was steady-state, and this set
        ss_last = np.isnan(t_last)
        ss_next = (t_solver is None) or np.isnan(t_solver[0])
        # If they're THE SAME, FUN3D will repeat the history
        if ss_last == ss_next:
            # Get last raw iteration reported by FUN3D
            iraw_last = self.get_lastrawiter()
            # Iterations to keep
            mask = i_solver > iraw_last
            # Trim them all
            for col in db:
                db[col] = db[col][mask]
            # Reset
            i_solver = db.get(dataBook.CASE_COL_ITRAW)
            t_solver = db.get(dataBook.CASE_COL_TRAW)
        # Get current last iter
        i_last = self.get_lastiter()
        # Copy to actual
        i_cape = i_solver.copy()
        # Required delta for iteration counter
        di = max(0, i_last - i_solver[0] + 1)
        # Modify history
        i_cape += di
        # Save iterations
        db.save_col(dataBook.CASE_COL_ITERS, i_cape)
        # Modify time history
        if (t_solver is None) or (t_solver[0] < 0):
            # No time histories
            t_raw = np.full(i_solver.size, np.nan)
            t_cape = np.full(i_solver.shape, np.nan)
            # Save placeholders for raw time
            db.save_col(dataBook.CASE_COL_TRAW, t_raw)
        else:
            # Get last time value
            t_last = self.get_maxtime()
            # Copy to actual
            t_cape = t_solver.copy()
            # Required delta for times to be ascending
            dt = max(0.0, np.floor(t_last - 2*t_solver[0] + t_solver[1]))
            # Modify time histories
            t_cape += dt
        # Save time histories
        db.save_col(dataBook.CASE_COL_TIME, t_cape)
        # Output the offsets
        return di

    # Plot R_1
    def PlotR1(self, **kw):
        r"""Plot the density

        :Call:
            >>> h = hist.PlotR1(n=None, nFirst=None, nLast=None, **kw)
        :Inputs:
            *hist*: :class:`cape.pyfun.dataBook.CaseResid`
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
            * 2015-10-21 ``@ddalle``: v1.0
        """
        # Plot "R_1"
        return self.PlotResid('R_1', YLabel='Density Residual', **kw)

    # Plot turbulence residual
    def PlotTurbResid(self, **kw):
        r"""Plot the turbulence residual

        :Call:
            >>> h = hist.PlotTurbResid(n=None, nFirst=None, nLast=None,
                    **kw)
        :Inputs:
            *hist*: :class:`cape.pyfun.dataBook.CaseResid`
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
            * 2015-10-21 ``@ddalle``: v1.0
        """
        # Plot "R_6"
        return self.PlotResid('R_6', YLabel='Turbulence Residual', **kw)

    # Read data from a second or later file
    def ReadFileAppend(self, fname):
        r"""Read data from a file and append it to current history

        :Call:
            >>> hist.ReadFileAppend(fname)
        :Inputs:
            *hist*: :class:`cape.pyfun.dataBook.CaseResid`
                Case force/moment history
            *fname*: :class:`str`
                Name of file to read
        :Versions:
            * 2016-05-05 ``@ddalle``: v1.0
            * 2016-10-28 ``@ddalle``: v1.1; catch iteration resets
            * 2024-01-11 ``@ddalle``: v1.2; DataKit updates
        """
        # Unpack iterations
        iters = self.get_values("i")
        # Process the column names
        nhdr, cols, inds = self.ProcessColumnNames(fname)
        # Check entries
        for col in cols:
            # Check for existing column
            if col in self:
                continue
            # Initialize the column
            self.save_col(col, np.zeros_like(iters, dtype="f8"))
        # Read the data.
        A = np.loadtxt(fname, skiprows=nhdr, usecols=tuple(inds))
        # Save current last iteration
        i1 = self["i"][-1]
        # Append the values.
        for k, col in enumerate(cols):
            # Column name
            V = A[:, k]
            # Check for iteration number reset
            if col == 'i' and V[0] < iters[-1]:
                # Keep counting iterations from the end of the previous one.
                V += (i1 - V[0] + 1)
            # Append
            self[col] = np.hstack((self[col], V))
        # Check for subiteration history
        Vsub = fname.split('.')
        fsub = Vsub[0][:-4] + "subhist." + (".".join(Vsub[1:]))
        # Check for the file
        if os.path.isfile(fsub):
            # Read the subiteration history
            self.ReadSubhist(fsub, iend=i1)
            return
        # Initialize residuals
        for k, col in enumerate(cols):
            # Get column name
            c0 = col + '0'
            # Check for special commands
            if not col.startswith('R'):
                continue
            # Copy the shape of the residual
            self[c0] = np.hstack((self[c0], np.nan*np.ones_like(V)))

    # Read subiteration history
    def ReadSubhist(self, fname, iend=0):
        r"""Read subiteration history

        :Call:
            >>> hist.ReadSubhist(fname)
        :Inputs:
            *hist*: :class:`cape.pyfun.dataBook.CaseResid`
                Fun3D residual history interface
            *fname*: :class:`str`
                Name of subiteration history file
            *iend*: {``0``} | positive :class:`int`
                Last iteration number before reading this file
        :Versions:
            * 2016-10-29 ``@ddalle``: v1.0
            * 2024-01-11 ``@ddalle``: v1.1; DataKit updates
        """
        # Initialize variables and read flag
        keys = []
        # Number of header lines
        nhdr = 0
        # Open the file
        f = open(fname)
        # Loop through lines
        while nhdr < 100:
            # Strip whitespace from the line.
            l = f.readline().strip()
            # Count line
            nhdr += 1
            # Check for "variables"
            if not l.lower().startswith('variables'):
                continue
            # Split on '=' sign.
            L = l.split('=')
            # Check for first variable.
            if len(L) < 2:
                break
            # Split variables on as things between quotes
            vals = re.findall(r'"[\w ]+"', L[1])
            # Append to the list.
            keys += [v.strip('"') for v in vals]
            break
        # Number of keys
        nkey = len(keys)
        # Read the data
        B = np.fromfile(f, sep=' ')
        # Get number of complete records
        nA = int(len(B) / nkey)
        # Reshape
        A = np.reshape(B[:nA*nkey], (nA, nkey))
        # Close the file
        f.close()
        # Initialize the output
        d = {}
        # Initialize column indices and their meanings.
        inds = []
        cols = []
        # Check for iteration column.
        if "Fractional_Time_Step" in keys:
            inds.append(keys.index("Fractional_Time_Step"))
            cols.append('i')
        # Check for residual of state 1
        if "R_1" in keys:
            inds.append(keys.index("R_1"))
            cols.append('R_1')
        # Check for residual of state 2
        if "R_2" in keys:
            inds.append(keys.index("R_2"))
            cols.append('R_2')
        # Check for residual of state 3
        if "R_3" in keys:
            inds.append(keys.index("R_3"))
            cols.append('R_3')
        # Check for residual of state 4
        if "R_4" in keys:
            inds.append(keys.index("R_4"))
            cols.append('R_4')
        # Check for residual of state 5
        if "R_5" in keys:
            inds.append(keys.index("R_5"))
            cols.append('R_5')
        # Check for turbulent residual
        if "R_6" in keys:
            inds.append(keys.index("R_6"))
            cols.append('R_6')
        if "R_7" in keys:
            inds.append(keys.index("R_7"))
            cols.append('R_7')
        # Loop through columns
        n = len(cols)
        for k in range(n):
            # Column name
            col = cols[k]
            # Save it
            d[col] = A[:, inds[k]]
        # Check for integers
        if 'i' not in d:
            return
        # Get iterations
        iters = self.get_values("i")
        # Indices of matching integers
        I = d['i'] == np.array(d['i'], dtype='int')
        # Don't read past the last write of '*_hist.dat'
        I = np.logical_and(I, d['i']+iend <= iters[-1])
        # Loop through the columns again to save them
        for k in range(n):
            # Column name
            col = cols[k]
            c0  = col + '0'
            # Get the values
            v = d[col][I]
            # Check integers
            if col == 'i':
                # Get expected iteration numbers
                ni = len(v)
                # Exit if no match
                # This happens when the subhist iterations have been written
                # but the corresponding iterations haven't been flushed yet.
                if ni == 0:
                    # No matches
                    ip = iters[0:0]
                else:
                    # Matches last *ni* iters
                    ip = iters[-ni:]
                # Offset current iteration numbers by reset iter
                iv = v + iend
                # Compare to existing iteration numbers
                if np.any(ip != iv):
                    print(
                        "Warning: Mismatch between nominal history " +
                        ("(%i-%i) and subiteration history (%i-%i)" %
                            (ip[0], ip[-1], iv[0], iv[-1])))
            # Check to append
            try:
                # Check if the attribute is present
                v0 = self[c0]
                # Get extra padding... again from missing subhist files
                n0 = iters.size - v0.size - v.size
                v1 = np.nan*np.ones(n0)
                # Save it if that command succeeded
                self[c0] = np.hstack((v0, v1, v))
            except KeyError:
                # Save the value as a new one
                self.save_col(c0, v)


# Function to fix iteration histories of one file
def _fix_iter(h: dataBook.CaseData, db: dict):
    r"""Fix iteration and time histories for FUN3D resets

    :Call:
        >>> _fix_iter(h, db)
    :Versions:
        * 2024-01-23 ``@ddalle``: v1.0
    """
    # Get iterations and time
    i_solver = db.get(dataBook.CASE_COL_ITRAW)
    t_solver = db.get(dataBook.CASE_COL_TRAW)
    # Check if we need to modify it
    if i_solver is not None:
        # Get current last iter
        i_last = h.get_lastiter()
        # Copy to actual
        i_cape = i_solver.copy()
        # Required delta for iteration counter
        di = max(0, i_last - i_solver[0] + 1)
        # Modify history
        i_cape += di
        # Save iterations
        db.save_col(dataBook.CASE_COL_ITERS, i_cape)
    # Modify time history
    if t_solver is None:
        # No time histories
        t_raw = np.full(i_solver.size, -1.0)
        t_cape = np.full(i_solver.shape, -1.0)
        # Save placeholders for raw time
        db.save_col(dataBook.CASE_COL_TRAW, t_raw)
    else:
        # Get last time value
        t_last = h.get_lasttime()
        # Copy to actual
        t_cape = t_solver.copy()
        # Required delta for times to be ascending
        dt = max(0.0, np.floor(t_last - 2*t_solver[0] + t_solver[1]))
        # Modify time histories
        t_cape += dt
    # Save time histories
    db.save_col(dataBook.CASE_COL_TIME, t_cape)
    # Output
    return db
