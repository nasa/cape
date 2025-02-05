r"""
:mod:`cape.pyfun.databook`: Post-processing for FUN3D data
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
book modules, :mod:`cape.cfdx.dataBook`, :mod:`cape.cfdx.lineload`, and
:mod:`cape.cfdx.pointsensor`.  However, some data book types may not be
implemented for all CFD solvers.

:See Also:
    * :mod:`cape.cfdx.dataBook`
    * :mod:`cape.cfdx.lineload`
    * :mod:`cape.cfdx.pointsensor`
    * :mod:`cape.pyfun.lineload`
    * :mod:`cape.options.databookopts`
"""

# Standard library modules
import os
import glob
import re

# Third-party modules
import numpy as np

# Local imports
from . import casecntl
from . import lineload
from . import pointsensor
from . import pltfile
from ..cfdx import databook
from ..dkit import tsvfile


# Radian -> degree conversion
deg = np.pi / 180.0

# Column names for FM files
COLNAMES_FM = {
    "Iteration": databook.CASE_COL_ITRAW,
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
    "p/p<sub>0</sub>": "phat",
    "p<sub>t</sub>/p<sub>0</sub>": "p0hat",
    "T<sub>t</sub>": "T0",
    "T<sub>RMS</sub>": "Trms",
    "Mach": "mach",
    "Simulation Time": databook.CASE_COL_TRAW,
}

# Column names for primary history, {PROJ}_hist.dat
COLNAMES_HIST = {
    "Iteration": databook.CASE_COL_ITRAW,
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
    "Simulation_Time": databook.CASE_COL_TRAW,
}

# Column names for fractional time step history, {PROJ}_subhist.dat
COLNAMES_SUBHIST = {
    "Fractional_Time_Step": databook.CASE_COL_SUB_ITRAW,
    "R_1": "R_1_sub",
    "R_2": "R_2_sub",
    "R_3": "R_3_sub",
    "R_4": "R_4_sub",
    "R_5": "R_5_sub",
    "R_6": "R_6_sub",
    "R_7": "R_7_sub",
    "C_x": "CA_sub",
    "C_y": "CY_sub",
    "C_z": "CN_sub",
    "C_M_x": "CLL_sub",
    "C_M_y": "CLM_sub",
    "C_M_z": "CLN_sub",
    "C_L": "CL_sub",
    "C_D": "CD_sub",
}


# Component data book
class DBFM(databook.DBFM):
    # Read case FM history
    def ReadCase(self, comp):
        r"""Read a :class:`CaseFM` object

        :Call:
            >>> FM = DB.ReadCaseFM(comp)
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *FM*: :class:`cape.pyfun.databook.CaseFM`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseFM(self.proj, comp)

    # Read case residual
    def ReadCaseResid(self):
        r"""Read a :class:`CaseResid` object

        :Call:
            >>> H = DB.ReadCaseResid()
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
        :Outputs:
            *H*: :class:`cape.pyfun.databook.CaseResid`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseResid(self.proj)


class DBProp(databook.DBProp):
    # Read case residual
    def ReadCaseResid(self):
        r"""Read a :class:`CaseResid` object

        :Call:
            >>> H = DB.ReadCaseResid()
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
        :Outputs:
            *H*: :class:`cape.pyfun.databook.CaseResid`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseResid(self.proj)


class DBPyFunc(databook.DBPyFunc):
    pass


# Data book target instance
class DBTarget(databook.DBTarget):
    pass


# TriqFM data book
class DBTriqFM(databook.DBTriqFM):
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
        *DBF*: :class:`cape.pyfun.databook.DBTriqFM`
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
            *DBF*: :class:`cape.pyfun.databook.DBTriqFM`
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
            * 2016-12-19 ``@ddalle``: v1.0
            * 2024-12-03 ``@ddalle``: v2.0; use ``CaseRunner`` method
        """
        # Get main retults
        ftriq, n, i0, i1 = casecntl.GetTriqFile()
        # Prepend that it was always found w/ new method
        return True, ftriq, n, i0, i1

    # Preprocess triq file (convert from PLT)
    def PreprocessTriq(self, ftriq, **kw):
        r"""Perform any necessary preprocessing to create ``triq`` file

        :Call:
            >>> DBL.PreprocessTriq(ftriq, i=None)
        :Inputs:
            *DBF*: :class:`cape.pyfun.databook.DBTriqFM`
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
            mach = casecntl.ReadConditions('mach')
        else:
            # Get from trajectory
            mach = self.x.GetMach(i)
        # Output format
        fmt = self.opts.get_DataBookTriqFormat(self.comp)
        # Read the plt information
        pltfile.Plt2Triq(fplt, ftriq, mach=mach, fmt=fmt)


class DBTriqFMComp(databook.DBTriqFMComp):
    pass


class DBTS(databook.DBTS):
    # Read case residual
    def ReadCaseResid(self):
        r"""Read a :class:`CaseResid` object

        :Call:
            >>> H = DB.ReadCaseResid()
        :Inputs:
            *DB*: :class:`cape.cfdx.databook.DataBook`
                Instance of data book class
        :Outputs:
            *H*: :class:`cape.pyfun.databook.CaseResid`
                Residual history class
        :Versions:
            * 2017-04-13 ``@ddalle``: First separate version
        """
        # Read CaseResid object from PWD
        return CaseResid(self.proj)


# Force/moment history
class CaseFM(databook.CaseFM):
    r"""Iterative force & moment histories for one case, one component

    This class contains methods for reading data about an the history
    of an individual component for a single casecntl.  It reads the Tecplot
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
        databook.CaseFM.__init__(self, comp, **kw)

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
        i_solver = db.get(databook.CASE_COL_ITRAW)
        t_solver = db.get(databook.CASE_COL_TRAW)
        # Get current last iter
        i_last = self.get_lastiter()
        # Copy to actual
        i_cape = i_solver.copy()
        # Required delta for iteration counter
        di = max(0, i_last - i_solver[0] + 1)
        # Modify history
        i_cape += di
        # Save iterations
        db.save_col(databook.CASE_COL_ITERS, i_cape)
        # Modify time history
        if (t_solver is None) or (t_solver[0] < 0):
            # No time histories
            t_raw = np.full(i_solver.size, np.nan)
            t_cape = np.full(i_solver.shape, np.nan)
            # Save placeholders for raw time
            db.save_col(databook.CASE_COL_TRAW, t_raw)
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
        db.save_col(databook.CASE_COL_TIME, t_cape)
        # Output
        return db


# Class to keep track of residuals
class CaseResid(databook.CaseResid):
    r"""FUN3D iterative history class

    This class provides an interface to residuals, CPU time, and
    similar data for a given case

    :Call:
        >>> hist = CaseResid(proj)
    :Inputs:
        *proj*: :class:`str`
            Project root name
    :Outputs:
        *hist*: :class:`cape.pyfun.databook.CaseResid`
            Instance of the run history class
    """
    # Default residual
    _default_resid = "L2Resid"
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
        'L2Resid_0'
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
        'L2Resid_0',
    )
    # This clkass expects subiterations
    _has_subiters = True

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
        databook.CaseResid.__init__(self, **kw)

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

    # Get list of files to read
    def get_subiter_filelist(self) -> list:
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
        fname = fw(f"{proj}_subhist.dat")
        fglob0a = fw(f"{proj}_subhist.old[0-9][0-9].dat")
        # Patters for multiple-file scenarios
        fglob1 = fw(f"{proj}_subhist.[0-9][0-9].dat")
        fglob2 = fw(f"{proj}[0-9][0-9]_subhist.dat")
        fglob3 = fw(f"{proj}[0-9][0-9]_subhist.[0-9][0-9].dat")
        fglob2a = fw(f"{proj}[0-9][0-9]_subhist.old[0-9][0-9].dat")
        # List of files
        filelist = []
        # Check which scenario we're in
        if os.path.isfile(fname):
            # Single project + original case; check for history resets
            glob1 = glob.glob(fglob1)
            glob0 = glob.glob(fglob0a)
            glob0.sort()
            glob1.sort()
            # Add in main file name
            filelist = glob1 + glob0 + [fname]
        else:
            # Multiple projects; try original case first
            glob2 = glob.glob(fglob2)
            glob3 = glob.glob(fglob3)
            glob2a = glob.glob(fglob2a)
            # Sort each
            glob2.sort()
            glob3.sort()
            glob2a.sort()
            # Combine both matches
            filelist = glob3 + glob2a + glob2
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
            >>> db = h.readfile(fname)
        :Inputs:
            *h*: :class:`CaseResid`
                Case residual history instance
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
        self._fix_iter(db)
        # Calculate L2 norms by adding up R_{n} contributions
        self._build_l2(db)
        # Output
        return db

    # Read subhistory file
    def readfile_subiter(self, fname: str) -> dict:
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
            * 2024-01-23 ``@ddalle``: v1.0 (readfile_subhist)
            * 2024-02-21 ``@ddalle``: v2.0
        """
        # Name of file used for base iterations
        fhist = fname.replace("subhist", "hist")
        fhist = re.sub(r"\.old[0-9][0-9]", "", fhist)
        # Check if previously read
        if fhist in self[databook.CASE_COL_NAMES]:
            # Get index of that file
            jsrc = self[databook.CASE_COL_NAMES].index(fhist)
            # Iteration history
            i = self[databook.CASE_COL_ITERS]
            # Index of which file each iteration came from
            itsrc = self[databook.CASE_COL_ITSRC]
            # Find iterations from that file, then previous files
            jsrc_mask, = np.where(itsrc == jsrc)
            jprev_mask, = np.where(itsrc < jsrc)
            # Check for matches
            if jsrc_mask.size > 0:
                # Use first iteration previously read from this file
                isrc = jsrc_mask[0]
                # Get raw iter and adjusted iter for first entry from that file
                i0 = i[isrc]
                i0raw = self[databook.CASE_COL_ITRAW][isrc]
                # Calculate offset
                di = i0 - i0raw
            elif jprev_mask.size > 0:
                # Offset to make new iter one greater than previous iter
                di = i[jprev_mask[-1]]
            else:
                # No adjustment to make
                di = 0
        else:
            di = 0
        # Read the _subhist.dat file
        db = tsvfile.TSVTecDatFile(fname, Translators=COLNAMES_SUBHIST)
        # Get the raw subiteration reported by FUN3D
        i_raw = db[databook.CASE_COL_SUB_ITRAW]
        # Modify iteration to global history value
        i_cape = i_raw + di
        # Save that
        db.save_col(databook.CASE_COL_SUB_ITERS, i_cape)
        # Build residual
        self._build_l2(db, suf="_sub")
        # Output
        return db

    # Calculate L2 norm(s)
    def _build_l2(self, db: tsvfile.TSVTecDatFile, suf=""):
        r"""Calculate *L2* norm for initial, final, and subiter

        :Versions:
            * 2024-01-24 ``@ddalle``: v1.0
        """
        # Column name for iteration
        icol = databook.CASE_COL_ITERS + suf
        # Column name for initial residual/value
        rcol = f"L2Resid{suf}"
        # Get iterations corresponding to this sufix
        iters = db.get(icol)
        # Skip if not present
        if iters is None:
            return
        # Initialize cumulative residual squared
        L2squared = np.zeros_like(iters, dtype="float")
        # Loop through potential residuals
        for c in ("R_1", "R_2", "R_3", "R_4", "R_5", "R_6", "R_7"):
            # Check for baseline
            col = c + suf
            # Get values
            v = db.get(col)
            # Assemble
            if v is not None:
                L2squared += v*v
        # Save residuals
        db.save_col(rcol, np.sqrt(L2squared))

    # Function to fix iteration histories of one file
    def _fix_iter(self, db: tsvfile.TSVTecDatFile) -> float:
        r"""Fix iteration and time histories for FUN3D resets

        :Call:
            >>> di = _fix_iter(h, db)
        :Outputs:
            *di*: :class:`float`
                Offset from FUN3D-reported iter to CAPE iter
        :Versions:
            * 2024-01-23 ``@ddalle``: v1.0
        """
        # Get last time step
        # (to check if we're switching time-accurate <-> steady-state)
        t_last = self.get_lasttime()
        # Get iterations and time
        i_solver = db.get(databook.CASE_COL_ITRAW)
        t_solver = db.get(databook.CASE_COL_TRAW)
        # Check if last reported iter was steady-state, and this set
        ss_last = np.isnan(t_last)
        ss_next = (t_solver is None) or np.isnan(t_solver[0])
        # If they're THE SAME, FUN3D will repeat the history
        if (ss_last == ss_next):
            # Get last raw iteration reported by FUN3D
            iraw_last = self.get_lastrawiter()
            # Iterations to keep
            mask = i_solver > iraw_last
            # Trim them all
            for col in db:
                db[col] = db[col][mask]
            # Reset
            i_solver = db.get(databook.CASE_COL_ITRAW)
            t_solver = db.get(databook.CASE_COL_TRAW)
        # Get current last iter
        i_last = self.get_lastiter()
        # Copy to actual
        i_cape = i_solver.copy()
        # Initial iter and time reported from solver
        i_solver0 = 0 if i_solver.size == 0 else i_solver[0]
        # Required delta for iteration counter
        di = max(0, i_last - i_solver0 + 1)
        # Modify history
        i_cape += di
        # Save iterations
        db.save_col(databook.CASE_COL_ITERS, i_cape)
        # Modify time history
        if (t_solver is None) or (t_solver.size == 0):
            # No time histories
            t_raw = np.full(i_solver.size, np.nan)
            t_cape = np.full(i_solver.shape, np.nan)
            # Save placeholders for raw time
            db.save_col(databook.CASE_COL_TRAW, t_raw)
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
        db.save_col(databook.CASE_COL_TIME, t_cape)
        # Output the offsets
        return di

    # Plot R_1
    def PlotR1(self, **kw):
        r"""Plot the density

        :Call:
            >>> h = hist.PlotR1(n=None, nFirst=None, nLast=None, **kw)
        :Inputs:
            *hist*: :class:`cape.pyfun.databook.CaseResid`
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
            *hist*: :class:`cape.pyfun.databook.CaseResid`
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


# Aerodynamic history class
class DataBook(databook.DataBook):
    r"""This class provides an interface to the data book for a given
    CFD run matrix.

    :Call:
        >>> DB = pyFun.databook.DataBook(x, opts)
    :Inputs:
        *x*: :class:`cape.pyfun.runmatrix.RunMatrix`
            The current pyFun trajectory (i.e. run matrix)
        *opts*: :class:`cape.pyfun.options.Options`
            Global pyFun options instance
    :Outputs:
        *DB*: :class:`cape.pyfun.databook.DataBook`
            Instance of the pyFun data book class
    """
    _fm_cls = DBFM
    _triqfm_cls = DBTriqFMComp
    _triqpt_cls = pointsensor.DBTriqPointGroup
    _ts_cls = DBTS
    _prop_cls = DBProp
    _pyfunc_cls = DBPyFunc
  # ===========
  # Readers
  # ===========

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
            self.LineLoads[comp] = lineload.DBLineLoad(
                comp, self.cntl,
                conf=conf, RootDir=self.RootDir, targ=self.targ)
        else:
            # Read as a specified target.
            ttl = '%s\\%s' % (targ, comp)
            # Get the keys
            topts = self.opts.get_DataBookTargetByName(targ)
            keys = topts.get("Keys", self.x.cols)
            # Read the file.
            self.LineLoads[ttl] = lineload.DBLineLoad(
                comp, self.cntl, keys=keys,
                conf=conf, RootDir=self.RootDir, targ=targ)

    # Read TriqPoint components
    def ReadTriqPoint(self, comp, check=False, lock=False, **kw):
        r"""Read a TriqPoint data book if not already present

        :Call:
            >>> DB.ReadTriqPoint(comp, check=False, lock=False, **kw)
        :Inputs:
            *DB*: :class:`cape.pyfun.databook.DataBook`
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
            self.TriqPoint[comp] = pointsensor.DBTriqPointGroup(
                self.cntl, self.opts, comp, pts=pts,
                RootDir=self.RootDir, check=check, lock=lock)
            # Return to starting position
            os.chdir(fpwd)

  # >

  # ========
  # Case I/O
  # ========
  # <
  # >


# Function to fix iteration histories of one file
def _fix_iter(h: databook.CaseData, db: dict):
    r"""Fix iteration and time histories for FUN3D resets

    :Call:
        >>> _fix_iter(h, db)
    :Versions:
        * 2024-01-23 ``@ddalle``: v1.0
    """
    # Get iterations and time
    i_solver = db.get(databook.CASE_COL_ITRAW)
    t_solver = db.get(databook.CASE_COL_TRAW)
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
        db.save_col(databook.CASE_COL_ITERS, i_cape)
    # Modify time history
    if t_solver is None:
        # No time histories
        t_raw = np.full(i_solver.size, -1.0)
        t_cape = np.full(i_solver.shape, -1.0)
        # Save placeholders for raw time
        db.save_col(databook.CASE_COL_TRAW, t_raw)
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
    db.save_col(databook.CASE_COL_TIME, t_cape)
    # Output
    return db
