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
import shutil

# Third-party modules
import numpy as np

# Local imports
from . import case
from . import lineLoad
from . import pointSensor
from . import plt
from ..cfdx import dataBook


# Radian -> degree conversion
deg = np.pi / 180.0


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
    def __init__(self, proj, comp):
        r"""Initialization method"""
        # Save component name
        self.comp = comp
        # Get the project rootname
        self.proj = proj
        # File names use lower case here
        compl = comp.lower()
        # Check for ``Flow`` folder
        if os.path.isdir('Flow'):
            # Dual setup
            qdual = True
            os.chdir('Flow')
        else:
            # Single folder
            qdual = False
        # Expected name of the component history file(s)
        fname = "%s_fm_%s.dat" % (proj, comp)
        fnamel = "%s_fm_%s.dat" % (proj, compl)
        # Patters for multiple-file scenarios
        fglob1 = "%s_fm_%s.[0-9][0-9].dat" % (proj, comp)
        fglob2 = "%s[0-9][0-9]_fm_%s.dat" % (proj, comp)
        fglob3 = "%s[0-9][0-9]_fm_%s.[0-9][0-9].dat" % (proj, comp)
        # Lower-case versions
        fglob1l = "%s_fm_%s.[0-9][0-9].dat" % (proj, compl)
        fglob2l = "%s[0-9][0-9]_fm_%s.dat" % (proj, compl)
        fglob3l = "%s[0-9][0-9]_fm_%s.[0-9][0-9].dat" % (proj, compl)
        # Check which scenario we're in
        if os.path.isfile(fname):
            # Save original version
            self.fname = fname
            # Single project + original case; check for history resets
            glob1 = glob.glob(fglob1)
            glob1.sort()
            # Add in main file name
            self.fglob = glob1 + [fname]
        elif os.path.isfile(fnamel):
            # Save lower-case version
            self.fname = fnamel
            # Single project + original case; check for history resets
            glob1 = glob.glob(fglob1l)
            glob1.sort()
            # Add in main file name
            self.fglob = glob1 + [fnamel]
        else:
            # Multiple projects; try original case first
            glob2 = glob.glob(fglob2)
            glob3 = glob.glob(fglob3)
            # Check for at least one match
            if len(glob2 + glob3) > 0:
                # Save original case
                self.fglob = glob2 + glob3
                self.fname = fname
            else:
                # Find lower-case matches
                glob2 = glob.glob(fglob2l)
                glob3 = glob.glob(fglob3l)
                # Save lower-case versions
                self.fglob = glob2 + glob3
                self.fname = fnamel
            # Sort whatever list we've god
            self.fglob.sort()
        # Check for available files.
        if len(self.fglob) > 0:
            # Read the first file
            self.ReadFileInit(self.fglob[0])
            # Loop through other files
            for fname in self.fglob[1:]:
                # Append the data
                self.ReadFileAppend(fname)
        else:
            # Make an empty CaseFM
            self.init_empty()
        # Return if necessary
        if qdual:
            os.chdir('..')

    # Read data from an initial file
    def ReadFileInit(self, fname=None):
        r"""Read data from a file and initialize columns

        :Call:
            >>> FM.ReadFileInit(fname=None)
        :Inputs:
            *FM*: :class:`cape.pyfun.dataBook.CaseFM`
                Case force/moment history
            *fname*: {``None``} | :class:`str`
                Name of file to process (defaults to *FM.fname*)
        :Versions:
            * 2016-05-05 ``@ddalle``: v1.0
            * 2023-01-11 ``@ddalle``: v1.1; DataKit updates
        """
        # Default file name
        if fname is None:
            fname = self.fname
        # Process the column names
        nhdr, cols, coeffs, inds = self.ProcessColumnNames(fname)
        # Save entries
        self._hdr = nhdr
        self.cols = cols
        self.coeffs = coeffs
        self.inds = inds
        # Read the data.
        try:
            # First attempt
            A = np.loadtxt(fname, skiprows=nhdr, usecols=tuple(inds))
        except Exception:
            # Copy to extra file
            fname1 = fname + '1'
            # Copy the file
            shutil.copy(fname, fname1)
            # Attempt to remove null characters
            try:
                os.system("sed -i 's/\\x0//g' %s" % fname1)
            except Exception:
                pass
            # Second attempt
            A = np.loadtxt(fname1, skiprows=nhdr, usecols=tuple(inds))
            # Remove copied file
            os.remove(fname1)
        # Save the values
        for k, col in enumerate(list(self.cols)):
            self.save_col(col, A[:, k])

    # Read data from a second or later file
    def ReadFileAppend(self, fname):
        r"""Read data from a file and append it to current history

        :Call:
            >>> FM.ReadFileAppend(fname)
        :Inputs:
            *FM*: :class:`cape.pyfun.dataBook.CaseFM`
                Case force/moment history
            *fname*: :class:`str`
                Name of file to read
        :Versions:
            * 2016-05-05 ``@ddalle``: v1.0
            * 2016-10-28 ``@ddalle``: v1.1; track iteration resets
            * 2023-01-11 ``@ddalle``: v1.2; DataKit updates
        """
        # Process the column names
        nhdr, cols, coeffs, inds = self.ProcessColumnNames(fname)
        # Get iterations
        iters = self.get_values("i")
        # Check entries
        for col in cols:
            # Check for existing column
            if col in self.cols:
                continue
            # Initialize the column
            self.save_col(col, np.zeros_like(iters, dtype="f8"))
            # Append to the end of the list
            self.cols.append(col)
        # Read the data.
        try:
            # First attempt
            A = np.loadtxt(fname, skiprows=nhdr, usecols=tuple(inds))
        except Exception:
            # Copy file and remove null chars
            fname1 = fname + '1'
            # Copy the file
            shutil.copy(fname, fname1)
            # Attempt to remove null characters
            try:
                os.system("sed -i 's/\\x0//g' %s" % fname1)
            except Exception:
                pass
            # Second attempt
            try:
                # Read the file
                A = np.loadtxt(fname1, skiprows=nhdr, usecols=tuple(inds))
                # Delete file
                os.remove(fname1)
            except Exception:
                # Status message
                print("Failed to read file '%s'" % fname)
                return
        # Save column data
        for k, col in enumerate(list(cols)):
            # Value to use
            V = A[:, k]
            # Check for iteration number reset
            if col == 'i' and V[0] < iters[-1]:
                # Keep counting iterations from the end of the previous one
                V += (iters[-1] - V[0] + 1)
            # Append
            self.save_col(col, np.hstack((self[col], V)))

    # Process the column names
    def ProcessColumnNames(self, fname=None):
        r"""Determine column names

        :Call:
            >>> nhdr, cols, coeffs, inds =
                        FM.ProcessColumnNames(fname=None)
        :Inputs:
            *FM*: :class:`cape.pyfun.dataBook.CaseFM`
                Case force/moment history
            *fname*: {``None``} | :class:`str`
                Name of file to process, defaults to *FM.fname*
        :Outputs:
            *nhdr*: :class:`int`
                Number of header rows to skip
            *cols*: :class:`list`\ [:class:`str`]
                List of column names
            *coeffs*: :class:`list`\ [:class:`str`]
                List of coefficient names
            *inds*: :class:`list`\ [:class:`int`]
                List of column indices for each entry of *cols*
        :Versions:
            * 2015-10-20 ``@ddalle``: v1.0
            * 2016-05-05 ``@ddalle``: v2.0
                - return results instead of saving to *FM*
        """
        # Initialize variables and read flag
        keys = []
        flag = 0
        # Default file name
        if fname is None:
            fname = self.fname
        # Number of header lines
        nhdr = 0
        # Open the file
        f = open(fname)
        # Loop through lines
        while nhdr < 100:
            # Strip whitespace from the line.
            l = f.readline().strip()
            # Check the line
            if flag == 0:
                # Count line
                nhdr += 1
                # Check for "variables"
                if not l.lower().startswith('variables'):
                    continue
                # Set the flag.
                flag = True
                # Split on '=' sign.
                L = l.split('=')
                # Check for first variable.
                if len(L) < 2:
                    continue
                # Split variables on as things between quotes
                vals = re.findall(r'"[\w ]+"', L[1])
                # Append to the list.
                keys += [v.strip('"') for v in vals]
            elif flag == 1:
                # Count line
                nhdr += 1
                # Reading more lines of variables
                if not l.startswith('"'):
                    # Done with variables; read extra headers
                    flag = 2
                    continue
                # Split variables on as things between quotes
                vals = re.findall(r'"[\w ]+"', l)
                # Append to the list.
                keys += [v.strip('"') for v in vals]
            else:
                # Check if it starts with an integer
                try:
                    # If it's an integer, stop reading lines.
                    float(l.split()[0])
                    break
                except Exception:
                    # Line starts with something else; continue
                    nhdr += 1
                    continue
        # Close the file
        f.close()
        # Initialize column indices and their meanings.
        inds = []
        cols = []
        coeffs = []
        # Check for iteration column.
        if "Iteration" in keys:
            inds.append(keys.index("Iteration"))
            cols.append('i')
        # Check for CA (axial force)
        if "C_x" in keys:
            inds.append(keys.index("C_x"))
            cols.append('CA')
            coeffs.append('CA')
        # Check for CY (body side force)
        if "C_y" in keys:
            inds.append(keys.index("C_y"))
            cols.append('CY')
            coeffs.append('CY')
        # Check for CN (normal force)
        if "C_z" in keys:
            inds.append(keys.index("C_z"))
            cols.append('CN')
            coeffs.append('CN')
        # Check for CLL (rolling moment)
        if "C_M_x" in keys:
            inds.append(keys.index("C_M_x"))
            cols.append('CLL')
            coeffs.append('CLL')
        # Check for CLM (pitching moment)
        if "C_M_y" in keys:
            inds.append(keys.index("C_M_y"))
            cols.append('CLM')
            coeffs.append('CLM')
        # Check for CLN (yawing moment)
        if "C_M_z" in keys:
            inds.append(keys.index("C_M_z"))
            cols.append('CLN')
            coeffs.append('CLN')
        # Check for CL
        if "C_L" in keys:
            inds.append(keys.index("C_L"))
            cols.append('CL')
            coeffs.append('CL')
        # Check for CD
        if "C_D" in keys:
            inds.append(keys.index("C_D"))
            cols.append('CD')
            coeffs.append('CD')
        # Check for CA (axial force)
        if "C_xp" in keys:
            inds.append(keys.index("C_xp"))
            cols.append('CAp')
            coeffs.append('CAp')
        # Check for CY (body side force)
        if "C_yp" in keys:
            inds.append(keys.index("C_yp"))
            cols.append('CYp')
            coeffs.append('CYp')
        # Check for CN (normal force)
        if "C_zp" in keys:
            inds.append(keys.index("C_zp"))
            cols.append('CNp')
            coeffs.append('CNp')
        # Check for CLL (rolling moment)
        if "C_M_xp" in keys:
            inds.append(keys.index("C_M_xp"))
            cols.append('CLLp')
            coeffs.append('CLLp')
        # Check for CLM (pitching moment)
        if "C_M_yp" in keys:
            inds.append(keys.index("C_M_yp"))
            cols.append('CLMp')
            coeffs.append('CLMp')
        # Check for CLN (yawing moment)
        if "C_M_zp" in keys:
            inds.append(keys.index("C_M_zp"))
            cols.append('CLNp')
            coeffs.append('CLNp')
        # Check for CL
        if "C_Lp" in keys:
            inds.append(keys.index("C_Lp"))
            cols.append('CLp')
            coeffs.append('CLp')
        # Check for CD
        if "C_Dp" in keys:
            inds.append(keys.index("C_Dp"))
            cols.append('CDp')
            coeffs.append('CDp')
        # Check for CA (axial force)
        if "C_xv" in keys:
            inds.append(keys.index("C_xv"))
            cols.append('CAv')
            coeffs.append('CAv')
        # Check for CY (body side force)
        if "C_yv" in keys:
            inds.append(keys.index("C_yv"))
            cols.append('CYv')
            coeffs.append('CYv')
        # Check for CN (normal force)
        if "C_zv" in keys:
            inds.append(keys.index("C_zv"))
            cols.append('CNv')
            coeffs.append('CNv')
        # Check for CLL (rolling moment)
        if "C_M_xv" in keys:
            inds.append(keys.index("C_M_xv"))
            cols.append('CLLv')
            coeffs.append('CLLv')
        # Check for CLM (pitching moment)
        if "C_M_yv" in keys:
            inds.append(keys.index("C_M_yv"))
            cols.append('CLMv')
            coeffs.append('CLMv')
        # Check for CLN (yawing moment)
        if "C_M_zv" in keys:
            inds.append(keys.index("C_M_zv"))
            cols.append('CLNv')
            coeffs.append('CLNv')
        # Check for CL
        if "C_Lv" in keys:
            inds.append(keys.index("C_Lv"))
            cols.append('CLv')
            coeffs.append('CLv')
        # Check for CD
        if "C_Dv" in keys:
            inds.append(keys.index("C_Dv"))
            cols.append('CDv')
            coeffs.append('CDv')
        # Check for mass flow
        if "Mass flow" in keys:
            inds.append(keys.index("Mass flow"))
            cols.append('mdot')
            coeffs.append('mdot')
        # Output
        return nhdr, cols, coeffs, inds


# Class to keep track of residuals
class CaseResid(dataBook.CaseResid):
    r"""FUN3D iterative history class

    This class provides an interface to residuals, CPU time, and
    similar data for a given case

    :Call:
        >>> hist = pyFun.dataBook.CaseResid(proj)
    :Inputs:
        *proj*: :class:`str`
            Project root name
    :Outputs:
        *hist*: :class:`cape.pyfun.dataBook.CaseResid`
            Instance of the run history class
    :Versions:
        * 2015-10-21 ``@ddalle``: v1.0
        * 2016-10-28 ``@ddalle``: v1.1; catch iteration resets
        * 2023-01-10 ``@ddalle``: v2.0; subclass to ``DataKit``
    """
    # Initialization method
    def __init__(self, proj):
        r"""Initialization method

        :Versions:
            * 2015-10-21 ``@ddalle``: v1.0
        """
        # Save the project root name
        self.proj = proj
        # Check for ``Flow`` folder
        if os.path.isdir('Flow'):
            # Dual setup
            qdual = True
            os.chdir('Flow')
        else:
            # Single folder
            qdual = False
        # Expected name of the history file
        self.fname = "%s_hist.dat" % proj
        # Full list
        if os.path.isfile(self.fname):
            # Single project; check for history resets
            fglob1 = glob.glob('%s_hist.[0-9][0-9].dat' % proj)
            fglob1.sort()
            # Add in main file name
            self.fglob = fglob1 + [self.fname]
        else:
            # Multiple adaptations
            fglob2 = glob.glob('%s[0-9][0-9]_hist.dat' % proj)
            fglob2.sort()
            # Check for history resets
            fglob1 = glob.glob('%s[0-9][0-9]_hist.[0-9][0-9].dat' % proj)
            fglob1.sort()
            # Combine history resets
            if len(fglob2) == 0:
                # Only have historical iterations right now
                self.fglob = fglob1
            elif len(fglob1) == 0:
                # We can use a single history file
                self.fglob = fglob2[-1:]
            else:
                # Get the adaption number from the last candidate of each glob
                nr = len(proj)
                na1 = int(fglob1[-1][nr:nr+2])
                na2 = int(fglob2[-1][nr:nr+2])
                # We need the pre-restart glob
                self.fglob = fglob1
                # Check if there is a newer active history file
                if na2 >= na1:
                    self.fglob.append(fglob2[-1])
        # Check for which file(s) to use
        if len(self.fglob) > 0:
            # Read the first file
            self.ReadFileInit(self.fglob[0])
            # Loop through other files
            for fname in self.fglob[1:]:
                # Append the data
                self.ReadFileAppend(fname)
        else:
            # Make an empty history
            self.init_empty()
        # Unpack iters
        iters = self.get_values("i")
        # Save number of iterations
        self.nIter = iters.size
        # Initialize residuals
        L2 = np.zeros(self.nIter)
        L0 = np.zeros(self.nIter)
        # Check residuals
        if 'R_1' in self.cols:
            L2 += (self["R_1"]**2)
        if 'R_2' in self.cols:
            L2 += (self["R_2"]**2)
        if 'R_3' in self.cols:
            L2 += (self["R_3"]**2)
        if 'R_4' in self.cols:
            L2 += (self["R_4"]**2)
        if 'R_5' in self.cols:
            L2 += (self["R_5"]**2)
        # Check initial subiteration residuals
        if 'R_10' in self.cols:
            L0 += (self["R_10"]**2)
        if 'R_20' in self.cols:
            L0 += (self["R_20"]**2)
        if 'R_30' in self.cols:
            L0 += (self["R_30"]**2)
        if 'R_40' in self.cols:
            L0 += (self["R_40"]**2)
        if 'R_50' in self.cols:
            L0 += (self["R_50"]**2)
        # Save residuals
        self.save_col("L2Resid", np.sqrt(L2))
        self.save_col("L2Resid0", np.sqrt(L0))
        # Return if appropriate
        if qdual:
            os.chdir('..')

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

    # Function to make empty one.
    def init_empty(self):
        r"""Create empty *CaseResid* instance

        :Call:
            >>> hist.init_empty()
        :Inputs:
            *hist*: :class:`cape.pyfun.dataBook.CaseResid`
                Case residual history
        :Versions:
            * 2015-10-20 ``@ddalle``: v1.0
            * 2024-01-11 ``@ddalle``: v1.1; DataKit updates
        """
        # Number of iterations
        self.nIter = 0
        # Save a default list of columns
        self.cols = [
            'i',
            'R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6', 'R_7',
            'L2Resid', 'L2Resid0'
        ]
        # Initialize
        for col in self.cols:
            self.save_col(col, np.zeros(0))

    # Process the column names
    def ProcessColumnNames(self, fname=None):
        r"""Determine column names

        :Call:
            >>> nhdr, cols, inds = hist.ProcessColumnNames(fname=None)
        :Inputs:
            *hist*: :class:`cape.pyfun.dataBook.CaseResid`
                Case force/moment history
            *fname*: {``None``} | :class:`str`
                File name to process, defaults to *FM.fname*
        :Outputs:
            *nhdr* :class:`int`
                Number of header rows
            *cols*: :class:`list`\ [:class:`str`]
                List of columns
            *inds*: :class:`list`\ [:class:`int`]
                List of indices in columns
        :Versions:
            * 2015-10-20 ``@ddalle``: v1.0
            * 2016-05-05 ``@ddalle``: Use output instead of saving to
                                      *FM*
        """
        # Default file name
        if fname is None:
            fname = self.fname
        # Initialize variables and read flag
        keys = []
        flag = 0
        # Number of header lines
        nhdr = 0
        # Open the file
        f = open(fname)
        # Loop through lines
        while nhdr < 100:
            # Strip whitespace from the line.
            l = f.readline().strip()
            # Check the line
            if flag == 0:
                # Count line
                nhdr += 1
                # Check for "variables"
                if not l.lower().startswith('variables'):
                    continue
                # Set the flag.
                flag = True
                # Split on '=' sign.
                L = l.split('=')
                # Check for first variable.
                if len(L) < 2:
                    continue
                # Split variables on as things between quotes
                vals = re.findall(r'"[\w ]+"', L[1])
                # Append to the list.
                keys += [v.strip('"') for v in vals]
            elif flag == 1:
                # Count line
                nhdr += 1
                # Reading more lines of variables
                if not l.startswith('"'):
                    # Done with variables; read extra headers
                    flag = 2
                    continue
                # Split variables on as things between quotes
                vals = re.findall(r'"[\w ]+"', l)
                # Append to the list.
                keys += [v.strip('"') for v in vals]
            else:
                # Check if it starts with an integer
                try:
                    # If it's an integer, stop reading lines.
                    float(l.split()[0])
                    break
                except Exception:
                    # Line starts with something else; continue
                    nhdr += 1
                    continue
        # Close the file
        f.close()
        # Initialize column indices and their meanings.
        inds = []
        cols = []
        # Check for iteration column.
        if "Iteration" in keys:
            inds.append(keys.index("Iteration"))
            cols.append('i')
        if "Fractional_Time_Step" in keys:
            inds.append(keys.index("Fractional_Time_Step"))
            cols.append('j')
        if "Wall Time" in keys:
            inds.append(keys.index("Wall Time"))
            cols.append('CPUtime')
        # Check for CA (axial force)
        if "R_1" in keys:
            inds.append(keys.index("R_1"))
            cols.append('R_1')
        # Check for CA (axial force)
        if "R_2" in keys:
            inds.append(keys.index("R_2"))
            cols.append('R_2')
        # Check for CA (axial force)
        if "R_3" in keys:
            inds.append(keys.index("R_3"))
            cols.append('R_3')
        # Check for CA (axial force)
        if "R_4" in keys:
            inds.append(keys.index("R_4"))
            cols.append('R_4')
        # Check for CA (axial force)
        if "R_5" in keys:
            inds.append(keys.index("R_5"))
            cols.append('R_5')
        # Check for CA (axial force)
        if "R_6" in keys:
            inds.append(keys.index("R_6"))
            cols.append('R_6')
        # Output
        if "R_7" in keys:
            inds.append(keys.index("R_7"))
            cols.append('R_7')
        return nhdr, cols, inds

    # Read initial data
    def ReadFileInit(self, fname=None):
        r"""Initialize history by reading a file

        :Call:
            >>> hist.ReadFileInit(fname=None)
        :Inputs:
            *hist*: :class:`cape.pyfun.dataBook.CaseResid`
                Case force/moment history
            *fname*: {``None``} | :class:`str`
                File name to process, defaults to *FM.fname*
        :Outputs:
            *nhdr* :class:`int`
                Number of header rows
            *cols*: :class:`list`\ [:class:`str`]
                List of columns
            *inds*: :class:`list`\ [:class:`int`]
                List of indices in columns
        :Versions:
            * 2015-10-20 ``@ddalle``: v1.0
            * 2016-05-05 ``@ddalle``: v1.1; return values
            * 2024-01-11 ``@ddalle``: v1.2; DataKit updates
        """
        # Default file name
        if fname is None:
            fname = self.fname
        # Process the column names
        nhdr, cols, inds = self.ProcessColumnNames(fname)
        # Save entries
        self._hdr = nhdr
        self.cols = cols
        self.inds = inds
        # Read the data
        A = np.loadtxt(fname, skiprows=nhdr, usecols=tuple(inds))
        # Save it
        for k, col in enumerate(list(self.cols)):
            self.save_col(col, A[:, k])
        # Check for subiteration history
        Vsub = fname.split('.')
        fsub = Vsub[0][:-40] + "subhist." + (".".join(Vsub[1:]))
        # Check for the file
        if os.path.isfile(fsub):
            # Process subiteration
            self.ReadSubhist(fsub)
            return
        # Initialize residuals
        for k, col in enumerate(list(cols)):
            # Get column name
            c0 = col + '0'
            # Check for special commands
            if not col.startswith('R'):
                continue
            # Copy the shape of the residual
            self.save_col(c0, np.nan*np.ones_like(self[col]))

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

    # Number of orders of magintude of residual drop
    def GetNOrders(self, nStats=1):
        r"""Get the number of orders of magnitude of residual drop

        :Call:
            >>> nOrders = hist.GetNOrders(nStats=1)

        :Inputs:
            *hist*: :class:`cape.cfdx.dataBook.CaseResid`
                Instance of the DataBook residual history
            *nStats*: :class:`int`
                Number of iterations to use for averaging the final
                residual
        :Outputs:
            *nOrders*: :class:`float`
                Number of orders of magnitude of residual drop
        :Versions:
            * 2015-10-21 ``@ddalle``: First version
            * 2024-01-11 ``@ddalle``: v1.2; DataKit updates
        """

        # Process the number of usable iterations available.
        i = max(self.nIter - nStats, 0)
        # Get the maximum residual.
        L1Max = np.log10(np.max(self["R_1"]))
        # Get the average terminal residual.
        L1End = np.log10(np.mean(self["R_1"][i:]))
        # Return the drop
        return L1Max - L1End

