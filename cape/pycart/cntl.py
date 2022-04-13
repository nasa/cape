"""
:mod:`cape.pycart.cntl`: Cart3D control module
===============================================

This module provides tools to quickly setup basic Cart3D runs or a complex
Cart3D setup from a small set of input files. Alternatively, the methods and
classes can be used to help setup a problem that is too complex or customized
to conform to standardized script libraries. A collection of cases combined
into a run matrix can be loaded using the following commands.

    .. code-block:: pycon
    
        >>> import cape.pycart.cart3d
        >>> cntl = cape.pycart.cntl.Cntl("pyCart.json")
        >>> cntl
        <cape.pycart.Cntl(nCase=4, tri='bullet.tri')>
        >>> cntl.x.GetFullFolderNames(0)
        'poweroff/m1.5a0.0b0.0'
        
        
An instance of this :class:`cape.pycart.cntl.Cntl` class has many
attributes, which include the run matrix (``cntl.x``), the options
interface (``cntl.opts``), and optionally the data book
(``cntl.DataBook``), the triangulation (``cntl.tri``), and the
appropriate input files (such as ``cntl.InputCntl``).

    ====================   =============================================
    Attribute              Class
    ====================   =============================================
    *cntl.x*               :class:`cape.runmatrix.RunMatrix`
    *cntl.opts*            :class:`cape.pycart.options.Options`
    *cntl.tri*             :class:`cape.tri.Tri`
    *cntl.DataBook*        :class:`cape.pycart.dataBook.DataBook`
    *cntl.InputCntl*       :class:`cape.pycart.inputCntl.InputCntl`
    *cntl.AeroCsh*         :class:`cape.pycart.aeroCsh.AeroCsh`
    ====================   =============================================

Finally, the :class:`cape.pycart.cntl.Cntl` class is subclassed from the
:class:`cape.cntl.Cntl` class, so any methods available to the CAPE class are
also available here.

"""

# Standard library modules
import os
import shutil
import json
import subprocess as sp
from datetime import datetime

# Third-party modules
import numpy as np

# Local imports
from . import options
from . import bin
from . import case
from . import manage
from . import dataBook
from . import report
from .. import cntl as capecntl
from ..cfdx import queue
from .inputCntl import InputCntl
from .aeroCsh import AeroCsh
from .preSpecCntl import PreSpecCntl
from .tri import Tri
from ..geom import RotatePoints
from ..runmatrix import RunMatrix

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyCartFolder = os.path.split(_fname)[0]
TemplateFodler = os.path.join(PyCartFolder, "templates")

    
# Class to read input files
class Cntl(capecntl.Cntl):
    """Class for handling global options and setup for Cart3D
    
    This class is intended to handle all settings used to describe a group
    of Cart3D cases. The settings are read from a JSON file.
    
    :Call:
        >>> cntl = cape.pycart.Cntl(fname="pyCart.json")
    :Inputs:
        *fname*: :class:`str`
            Name of pyCart input file
    :Outputs:
        *cntl*: :class:`cape.pycart.cntl.Cntl`
            Instance of the pyCart control class
    :Data members:
        *cntl.opts*: :class:`dict`
            Dictionary of options for this case (directly from *fname*)
        *cntl.x*: :class:`cape.runmatrix.RunMatrix`
            Values and definitions for variables in the run matrix
        *cntl.RootDir*: :class:`str`
            Absolute path to the root directory
    :Versions:
        * 2014-05-28 ``@ddalle``  : First version
        * 2014-06-03 ``@ddalle``  : Renamed class `Cntl` --> `Cart3d`
        * 2014-06-30 ``@ddalle``  : Reduced number of data members
        * 2014-07-27 ``@ddalle``  : `cart3d.RunMatrix` --> `cart3d.x`
    """
  # =============
  # Configuration
  # =============
  # <
    # Initialization method
    def __init__(self, fname="pyCart.json"):
        """Initialization method for :mod:`cape.cntl.Cntl`"""
        # Force default
        if fname is None:
            fname = "pyCart.json"
        # Check if file exists
        if not os.path.isfile(fname):
            # Raise error but without traceback
            os.sys.tracebacklimit = 0
            raise ValueError("No pyCart control file '%s' found" % fname)
        
        # Read settings
        self.opts = options.Options(fname=fname)
        
        #Save the current directory as the root
        self.RootDir = os.getcwd()
        
        # Import modules
        self.modules = {}
        self.ImportModules()
        
        # Process the trajectory.
        self.x = RunMatrix(**self.opts['RunMatrix'])

        # Job list
        self.jobs = {}
        
        # Set umask
        os.umask(self.opts.get_umask())
        
        # Run any initialization functions
        self.InitFunction()
        
    # Output representation
    def __repr__(self):
        """Output representation for the class."""
        # Display basic information from all three areas.
        return "<cape.pycart.Cntl(nCase=%i, tri='%s')>" % (
            self.x.nCase,
            self.opts.get_TriFile())
        
  # >
  
  # =======================
  # Command-Line Interface
  # =======================
  # <
    # Baseline function
    def cli(self, *a, **kw):
        """Command-line interface
        
        :Call:
            >>> cntl.cli(*a, **kw)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of control class containing relevant parameters
            *kw*: :class:`dict` (``True`` | ``False`` | :class:`str`)
                Unprocessed keyword arguments
        :Outputs:
            *cmd*: ``None`` | :class:`str`
                Name of command that was processed, if any
        :Versions:
            * 2018-10-19 ``@ddalle``: Content from ``bin/`` executables
        """
        # Preprocess command-line inputs
        a, kw = self.cli_preprocess(*a, **kw)
        # Call the common interface
        cmd = self.cli_cape(*a, **kw)
        # Test for a command
        if cmd is not None:
            return
        # Otherwise fall back to code-specific commands
        if kw.get('pt'):
            # Cart3D point sensor
            self.UpdatePointSensor(**kw)
        elif kw.get('a'):
            # Archive folders
            self.TarAdapt(**kw)
            self.TarViz(**kw)
        elif kw.get('explode'):
            # Break out each named component.
            self.ExplodeTri()
        else:
            # Submit the jobs
            self.SubmitJobs(**kw)
    
  # >
    
  # ==========
  # File Input
  # ==========
  # <
    # Function to read the databook.
    def ReadDataBook(self, comp=None):
        """Read the current data book
        
        :Call:
            >>> cntl.ReadDataBook(comp=None)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of control class containing relevant parameters
            *comp*: {``None``} | :class:`str` | :class:`list`
                List of components, or read all if ``None``
        :Versions:
            * 2014-12-28 ``@ddalle``: First version
            * 2017-04-27 ``@ddalle``: Added *comp* option
        """
        # Test for an existing data book.
        try:
            self.DataBook
            return
        except AttributeError:
            pass
        # Go to root directory.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Ensure list of components
        if comp is not None:
            comp = list(np.array(comp).flatten())
        # Read the data book.
        self.DataBook = dataBook.DataBook(self, comp=comp)
        # Return to original folder.
        os.chdir(fpwd)
        
    # Function to read a report
    def ReadReport(self, rep):
        """Read a report interface
        
        :Call:
            >>> R = cntl.ReadReport(rep)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of control class containing relevant parameters
            *rep*: :class:`str`
                Name of report
        :Outputs:
            *R*: :class:`pyFun.report.Report`
                Report interface
        :Versions:
            * 2018-10-19 ``@ddalle``: First version
        """
        # Read the report
        R = report.Report(self, rep)
        # Output
        return R
        
  # >
    
  # ==============
  # Case Interface
  # ==============
  # <
    # Read run control options from case JSON file
    def ReadCaseJSON(self, i):
        """Read ``case.json`` file from case *i* if possible
        
        :Call:
            >>> rc = cntl.ReadCaseJSON(i)
        :Inputs:
            *ofl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of FUN3D control class
            *i*: :class:`int`
                Run index
        :Outputs:
            *rc*: ``None`` | :class:`pyOver.options.runControl.RunControl`
                Run control interface read from ``case.json`` file
        :Versions:
            * 2016-12-12 ``@ddalle``: First version
            * 2017-03-31 ``@ddalle``: Copied from :mod:`cape.pyover`
        """
        # Safely go to root directory.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Check if it exists.
        if not os.path.isdir(frun):
            # Go back and quit.
            os.chdir(fpwd)
            return
        # Go to the folder.
        os.chdir(frun)
        # Check for file
        if not os.path.isfile('case.json'):
            # Nothing to read
            rc = None
        else:
            # Read the file
            rc = case.ReadCaseJSON()
        # Return to original location
        os.chdir(fpwd)
        # Output
        return rc
        
    # Call the correct :mod:`case` module
    def CaseStartCase(self):
        """Start a case by either submitting it or running it
        
        This function relies on :mod:`cape.pycart.case`, and so it is customized for
        the Cart3D solver only in that it calles the correct *case* module.
        
        :Call:
            >>> pbs = cntl.CaseStartCase()
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of control class containing relevant parameters
        :Outputs:
            *pbs*: :class:`int` or ``None``
                PBS job ID if submitted successfully
        :Versions:
            * 2015-10-14 ``@ddalle``: First version
        """
        return case.StartCase()
        
  # >
    
  # ===========
  # Case Status
  # ===========
  # <
        
    # Get the current iteration number from :mod:`case`
    def CaseGetCurrentIter(self):
        """Get the current iteration number from the appropriate module
        
        This function utilizes the :mod:`cape.case` module, and so it must be
        copied to the definition for each solver's control class
        
        :Call:
            >>> n = cntl.CaseGetCurrentIter()
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *n*: :class:`int` or ``None``
                Number of completed iterations or ``None`` if not set up
        :Versions:
            * 2015-10-14 ``@ddalle``: First version
        """
        return case.GetCurrentIter()
        
    # Get the current iteration number from :mod:`case`
    def CaseGetCurrentPhase(self):
        """Get the current phase number from the appropriate module
        
        This function utilizes the :mod:`cape.case` module, and so it must be
        copied to the definition for each solver's control class
        
        :Call:
            >>> j = cntl.CaseGetCurrentPhase()
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2017-06-29 ``@ddalle``: First version
        """
        # Be safe
        try:
            # Read the "case.json" folder
            rc = case.ReadCaseJSON()
            # Get the phase number
            return case.GetPhaseNumber(rc)
        except:
            return 0
            
    # Function to check if the mesh for case i exists
    def CheckMesh(self, i):
        """Check if the mesh for case *i* is prepared.
        
        :Call:
            >>> q = cntl.CheckMesh(i)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *q*: :class:`bool`
                Whether or not the mesh for case *i* is prepared
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
        """
        # Check input.
        if not type(i).__name__.startswith("int"):
            raise TypeError(
                "Input to :func:`Cart3d.CheckMesh()` must be :class:`int`.")
        # Get the group name.
        fgrp = self.x.GetGroupFolderNames(i)
        # Initialize with "pass" setting.
        q = True
        # Remember current location.
        fpwd = os.getcwd()
        # Go to root folder.
        os.chdir(self.RootDir)
        # Check if the folder exists.
        if (not os.path.isdir(fgrp)):
            os.chdir(fpwd)
            return False
        # Go to the group folder.
        os.chdir(fgrp)
        # Extract options
        opts = self.opts
        # Check for group mesh.
        if not opts.get_GroupMesh():
            # Get the case name.
            frun = self.x.GetFolderNames(i)
            # Check if it's there.
            if (not os.path.isdir(frun)):
                os.chdir(fpwd)
                return False
            # Go to the folder.
            os.chdir(frun)
        # Go to working folder. ('.' or 'adapt??/')
        os.chdir(case.GetWorkingFolder())
        # Check for a mesh file?
        if opts.get_GroupMesh() or opts.get_PreMesh(0):
            # Intersected mesh file.
            if not os.path.isfile('Components.i.tri'): q = False
            # Check for jumpstart exception
            if not opts.get_Adaptive(0) or opts.get_jumpstart(0):
                # Mesh file.
                if q and opts.get_mg() > 0:
                    # Look for multigrid mesh
                    if not os.path.isfile('Mesh.mg.c3d'): q = False
                elif q:
                    # Look for original mesh
                    if not os.path.isfile('Mesh.c3d'): q = False
        elif opts.get_intersect():
            # Pre-intersect surface files.
            if not os.path.isfile('Components.c.tri'): q = False
            if q and not os.path.isfile('Components.tri'): q = False
        else:
            # Intersected file
            if not os.path.isfile('Components.i.tri'): q = False
        # Return to original folder.
        os.chdir(fpwd)
        # Output.
        return q
        
    # Check if cases with zero iterations are not yet setup to run
    def CheckNone(self, v=False):
        """Check if the current folder has the necessary files to run
        
        :Call:
            >>> q = cntl.CheckNone(v=False)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of control class containing relevant parameters
            *v*: ``True`` | {``False``}
                Verbose flag; prints message if *q* is ``True``
        :Versions:
            * 2015-09-27 ``@ddalle``: First version
            * 2017-02-22 ``@ddalle``: Added verbose flag
        """
        # Check for the surface file.
        if not (os.path.isfile('Components.i.tri')
                or os.path.isfile('Components.tri')):
            if v: print("    Missing tri file")
            return True
        # Input file.
        if not os.path.isfile('input.00.cntl'):
            if v: print("    Missing file 'input.00.cntl'")
            return True
        # Settings file.
        if not os.path.isfile('case.json'):
            if v: print("    Missing file 'case.json'")
            return True
        # Read the settings.
        rc = case.ReadCaseJSON()
        # Check for which mesh file to look for.
        if rc.get_Adaptive(0):
            # Mesh file is gone or will be created during aero.csh
            pass
        elif not rc.get_PreMesh(0):
            # Mesh may be generated later.
            pass
        elif self.opts.get_mg() > 0:
            # Look for the multigrid mesh
            if not os.path.isfile('Mesh.mg.c3d'):
                # Check verbose flag
                if v: print("    Missing file 'Mesh.mg.c3d'")
                return True
        else:
            # Look for the original mesh
            if not os.path.isfile('Mesh.c3d'):
                # Check verbose flag
                if v: print("    Missing file 'Mesh.c3d'")
                return True
        # Apparently no issues.
        return False
   
  # >
    
  # =========
  # CPU Stats
  # =========
  # <
    # Get total CPU hours (actually core hours)
    def GetCPUTime(self, i, running=False):
        """Read a CAPE-style core-hour file from a case
        
        :Call:
            >>> CPUt = cntl.GetCPUTime(i, running=False)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Cart3D control interface
            *i*: :class:`int`
                Case index
            *running*: ``True`` | {``False``}
                Whether or not to check for time since last start
        :Outputs:
            *CPUt*: :class:`float` | ``None``
                Total core hours used in this job
        :Versions:
            * 2015-12-22 ``@ddalle``: First version
            * 2016-08-31 ``@ddalle``: Checking time since most recent start
        """
        # File names
        fname = 'pycart_time.dat'
        fstrt = 'pycart_start.dat'
        # Call the general function using hard-coded file name
        return self.GetCPUTimeBoth(i, fname, fstrt, running=running)
        
  # >
    
  # ================
  # Case Preparation
  # ================
  # <
    
   # +++++++
   # General
   # +++++++
   # [
    # Prepare a case.
    def PrepareCase(self, i):
        """Prepare case for running if necessary
        
        :Call:
            >>> cntl.PrepareCase(i)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of case to analyze
        :Versions:
            * 2014-09-30 ``@ddalle``: First version
        """
        # Get the existing status.
        n = self.CheckCase(i)
        # Quit if prepared.
        if n is not None: return None
        # Save current location.
        fpwd = os.getcwd()
        # Go to root folder.
        os.chdir(self.RootDir)
        # Case function
        self.CaseFunction(i)
        # Prepare the mesh.
        self.PrepareMesh(i)
        # Get the run name.
        frun = self.x.GetFullFolderNames(i)
        # Check for the run directory.
        if not os.path.isdir(frun): self.mkdir(frun)
        # Go there.
        os.chdir(frun)
        # Write the conditions to a simple JSON file.
        self.x.WriteConditionsJSON(i)
        # Different processes for GroupMesh and CaseMesh
        if self.opts.get_GroupMesh():
            # Copy the required files.
            for fname in ['input.c3d', 'preSpec.c3d.cntl', 
                    'Mesh.c3d.Info', 'Config.xml']:
                # Source path.
                fsrc = os.path.join(os.path.abspath('..'), fname)
                # Check for the file.
                if os.path.isfile(fsrc):
                    # Copy it.
                    shutil.copy(fsrc, fname)
            # Create links that are available.
            for fname in ['Mesh.c3d', 'Mesh.mg.c3d', 'Mesh.R.c3d',
                    'Components.i.tri', 'Components.tri', 'Components.c.tri']:
                # Source path.
                fsrc = os.path.join(os.path.abspath('..'), fname)
                # Remove the file if it's present.
                if os.path.isfile(fname):
                    os.remove(fname)
                # Check for the file.
                if os.path.isfile(fsrc):
                    # Create a symlink.
                    os.symlink(fsrc, fname)
        else:
            # Get the name of the configuration and input files.
            fxml = os.path.join(self.RootDir, self.opts.get_ConfigFile())
            fc3d = os.path.join(self.RootDir, self.opts.get_inputC3d())
            # Copy the config file.
            if os.path.isfile(fxml):
                shutil.copy(fxml, 'Config.xml')
            # Copy the input.c3d file.
            if os.path.isfile(fc3d):
                shutil.copy(fc3d, 'input.c3d')
        # Get function for setting boundary conditions, etc.
        keys = self.x.GetKeysByType('CaseFunction')
        # Get the list of functions.
        funcs = [self.x.defns[key]['Function'] for key in keys] 
        # Reread the input file(s).
        self.ReadInputCntl()
        self.ReadAeroCsh()
        # Loop through the functions.
        for (key, func) in zip(keys, funcs):
            # Apply it.
            exec("self.%s(self,%s,i=%i)" % (func, self.x[key][i], i))
        # Write the input.cntl and aero.csh file(s).
        self.PrepareInputCntl(i)
        self.PrepareAeroCsh(i)
        # Write a JSON files with flowCart and plot settings.
        self.WriteCaseJSON(i)
        # Write the PBS script.
        self.WritePBS(i)
        # Return to original location.
        os.chdir(fpwd)
  
   # ]
   
   # ++++
   # Mesh
   # ++++
   # [
    # Prepare the mesh for case i (if necessary)
    def PrepareMesh(self, i):
        """Prepare the mesh for case *i* if necessary.
        
        :Call:
            >>> cntl.PrepareMesh(i)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2014-09-29 ``@ddalle``: First version
        """
        # ---------
        # Case info
        # ---------
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Get name of group.
        fgrp = self.x.GetGroupFolderNames(i)
        # Check the mesh.
        if self.CheckMesh(i):
            return None
        # ------------------
        # Folder preparation
        # ------------------
        # Remember current location.
        fpwd = os.getcwd()
        # Go to root folder.
        os.chdir(self.RootDir)
        # Check for the group folder and make it if necessary.
        if not os.path.isdir(fgrp):
            self.mkdir(fgrp)
        # Check for groups with common meshes
        if self.opts.get_GroupMesh():
            # Get the group index.
            j = self.x.GetGroupIndex(i)
            # Status update
            print("  Group name: '%s' (index %i)" % (fgrp,j))
            # Go there.
            os.chdir(fgrp)
        else:
            # Check if the run folder exists.
            if not os.path.isdir(frun):
                self.mkdir(frun)
            # Status update.
            print("  Case name: '%s' (index %i)" % (frun,i))
            # Go there.
            os.chdir(frun)
        # ----------
        # Copy files
        # ----------
        # Get the name of the configuration file.
        fxml = os.path.join(self.RootDir, self.opts.get_ConfigFile())
        fpre = os.path.join(self.RootDir, self.opts.get_preSpecCntl())
        fc3d = os.path.join(self.RootDir, self.opts.get_inputC3d())
        # Copy the config file.
        if os.path.isfile(fxml):
            shutil.copyfile(fxml, 'Config.xml')
        # Copy the preSpec file.
        if os.path.isfile(fpre):
            shutil.copyfile(fpre, 'preSpec.c3d.cntl')
        # Copy the cubes input file.
        if os.path.isfile(fc3d):
            shutil.copyfile(fc3d, 'input.c3d')
        # ------------------
        # Triangulation prep
        # ------------------
        # Status update
        print("  Preparing surface triangulation...")
        # Read the mesh.
        self.ReadTri()
        # Revert to initial surface.
        self.tri = self.tri0.Copy()
        # Apply rotations, translations, etc.
        self.PrepareTri(i)
        # Check intersection status.
        if self.opts.get_intersect():
            # Write the tri file as non-intersected; each volume is one CompID
            self.tri.WriteVolTri('Components.tri')
            # Write the existing triangulation with existing CompIDs.
            self.tri.Write('Components.c.tri')
        else:
            # Write the tri file.
            self.tri.Write('Components.i.tri')
        # --------------------
        # Volume mesh creation
        # --------------------
        # Get functions for mesh functions.
        keys = self.x.GetKeysByType('MeshFunction')
        # Loop through the mesh functions
        for key in keys:
            # Get the function for this *MeshFunction*
            func = self.x.defns[key]['Function']
            # Apply it.
            exec("%s(self,%s,i=%i)" % (func, self.x[key][i], i))
        # RunControl options (for consistency)
        rc = self.opts['RunControl']
        # Run autoInputs if necessary.
        if self.opts.get_PreMesh(0) or not os.path.isfile('preSpec.c3d.cntl'):
            # Run autoInputs (tests opts.get_autoInputs() internally)
            case.CaseAutoInputs(rc, j=0)
        # Read the resulting preSpec.c3d.cntl file
        self.PreSpecCntl = PreSpecCntl('preSpec.c3d.cntl')
        # Bounding box control...
        self.PreparePreSpecCntl()
        # Check for jumpstart.
        if self.opts.get_PreMesh(0) or self.opts.get_GroupMesh():
            # Run ``intersect`` if appropriate
            case.CaseIntersect(rc)
            # Run ``verify`` if appropriate
            case.CaseVerify(rc)
            # Create the mesh if appropriate
            case.CaseCubes(rc, j=0)
        # Return to original folder.
        os.chdir(fpwd)
   # ]
    
   # ++++++++++++++++
   # preSpec.c3d.cntl
   # ++++++++++++++++
   # [
    # Function to prepare "input.cntl" files
    def PreparePreSpecCntl(self):
        """
        Prepare and write :file:`preSpec.c3d.cntl` according to the current
        settings and in the current folder.
        
        :Call:
            >>> cntl.PreparePreSpecCntl()
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of global pyCart settings object
        :See Also:
            * :func:`cape.pycart.options.Mesh.Mesh.get_BBox`
            * :func:`cape.pycart.options.Mesh.Mesh.get_XLev`
            * :func:`cape.tri.TriBase.GetCompBBox`
            * :func:`cape.pycart.preSpecCntl.PreSpecCntl.AddBBox`
            * :func:`cape.pycart.preSpecCntl.PreSpecCntl.AddXLev`
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Loop through BBoxes
        for BBox in self.opts.get_BBox():
            # Safely get number of refinements.
            n = BBox.get("n", 7)
            # Filter the type.
            if "compID" in BBox:
                # Bounding box specified relative to a component.
                xlim = self.tri.GetCompBBox(**BBox)
            else:
                # Bounding box coordinates given.
                xlim = BBox.get("xlim")
            # Check for degeneracy.
            if (not n) or (xlim is None): continue
            # Add the bounding box.
            self.PreSpecCntl.AddBBox(n, xlim)
        # Loop through the XLevs
        for XLev in self.opts.get_XLev():
            # Safely extract info from the XLev.
            n = XLev.get("n", 0)
            compID = XLev.get("compID", [])
            # Process it into a list of integers (if not already).
            compID = self.tri.config.GetCompID(compID)
            # Check for degeneracy.
            if (not n) or (not compID): continue
            # Add an XLev line.
            self.PreSpecCntl.AddXLev(n, compID)
        # Write the file.
        self.PreSpecCntl.Write('preSpec.c3d.cntl')
   # ]
    
   # ++++++++++
   # input.cntl
   # ++++++++++
   # [
    # Function to read the "input.cntl" file
    def ReadInputCntl(self):
        """Read the :file:`input.cntl` file

        :Call:
            >>> cntl.ReadInputCntl()
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of global pyCart settings object
        :Versions:
            * 2015-06-13 ``@ddalle``: First version
        """
        # Change to root safely.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Input file name
        fname = self.opts.get_InputCntl()
        # Check if the file exists.
        if os.path.isfile(fname):
            # Read the file.
            self.InputCntl = InputCntl(fname)
        else:
            # Use the template
            print("Using template for 'input.cntl' file")
            self.InputCntl = InputCntl(options.getCart3DTemplate('input.cntl'))
        # Go back to original location
        os.chdir(fpwd)
        
    # Function to prepare "input.cntl" files
    def PrepareInputCntl(self, i):
        """
        Write :file:`input.cntl` for run case *i* in the appropriate folder
        and with the appropriate settings.
        
        :Call:
            >>> cntl.PrepareInputCntl(i)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of global pyCart settings object
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014-06-04 ``@ddalle``: First version
            * 2014-06-06 ``@ddalle``: Low-level functionality for grid folders
            * 2014-09-30 ``@ddalle``: Changed to write only a single case
        """
        # Extract trajectory.
        x = self.x
        # Process the key types.
        KeyTypes = [x.defns[k]['Type'] for k in x.cols]
        # Go safely to root folder.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        
        # Set the flight conditions.
        # Mach number
        M = x.GetMach(i)
        if M is not None: self.InputCntl.SetMach(M)
        # Angle of attack
        a = x.GetAlpha(i)
        if a is not None: self.InputCntl.SetAlpha(a)
        # Sideslip angle
        b = x.GetBeta(i)
        if b is not None: self.InputCntl.SetBeta(b)
        
        # Specify list of forces to track with `clic`
        self.InputCntl.RequestForce(self.opts.get_ClicForces())
        # Set reference values.
        self.InputCntl.SetReferenceArea(self.opts.get_RefArea())
        self.InputCntl.SetReferenceLength(self.opts.get_RefLength())
        self.InputCntl.SetMomentPoint(self.opts.get_RefPoint())
        # Get the case.
        frun = self.x.GetFullFolderNames(i)
        # Make folder if necessary.
        if not os.path.isdir(frun): self.mkdir(frun)
        # Get the cut planes.
        XSlices = self.opts.get_Xslices()
        YSlices = self.opts.get_Yslices()
        ZSlices = self.opts.get_Zslices()
        # Process cut planes
        if XSlices: self.InputCntl.SetXSlices(XSlices)
        if YSlices: self.InputCntl.SetYSlices(YSlices)
        if ZSlices: self.InputCntl.SetZSlices(ZSlices)
        # Get the sensors
        PS = self.opts.get_PointSensors()
        LS = self.opts.get_LineSensors()
        # Process sensors
        if PS: self.InputCntl.SetPointSensors(PS)
        if LS: self.InputCntl.SetLineSensors(LS)
        # Loop through the output functional 'optForce's
        for Name, kw in self.opts.get_optForces().items():
            # Set the force.
            self.InputCntl.SetOutputForce(Name, **kw)
        # Loop through the output functional 'optSensor's
        for Name, kw in self.opts.get_optSensors().items():
            # Set the sensor.
            self.InputCntl.SetOutputSensor(Name, **kw)
        # Loop through the output functional 'optMoment's
        for Name, kw in self.opts.get_optMoments().items():
            # Set the sensor.
            self.InputCntl.SetOutputMoment(Name, **kw)
            
        # SurfBC keys
        for k in self.x.GetKeysByType('SurfBC'):
            # Apply the method
            self.SetSurfBC(k, i, CT=False)
        # SurfCT keys
        for k in self.x.GetKeysByType('SurfCT'):
            # Apply the method with the *CT* flag
            self.SetSurfBC(k, i, CT=True)
        
        # Loop through the phases.
        for j in range(self.opts.get_nSeq()):
            # Set up the Runge-Kutta coefficients.
            self.InputCntl.SetRungeKutta(self.opts.get_RKScheme(j))
            # Set the CFL number
            self.InputCntl.SetCFL(self.opts.get_cfl(j))
            # Write the number of orders of magnitude for early convergence.
            self.InputCntl.SetNOrders(self.opts.get_nOrders(j))
            # Get the first-order status.
            fo = self.opts.get_first_order(j)
            # Set the status.
            if fo:
                # Run `flowCart` in first-order mode (everywhere)
                self.InputCntl.SetFirstOrder()
            # Get robust mode.
            if self.opts.get_robust_mode(j):
                # Set robust mode.
                self.InputCntl.SetRobustMode()
            # Name of output file.
            fout = os.path.join(frun, 'input.%02i.cntl' % j)
            # Write the input file.
            self.InputCntl.Write(fout)
        # Return to original path.
        os.chdir(fpwd)
   # ]
    
   # ++++++
   # Thrust
   # ++++++
   # [
    # Function to get surface BC stuff
    def GetSurfBCState(self, key, i):
        """Get surface boundary condition state
        
        :Call:
            >>> rho, U, p = cntl.GetSurfBCState(key, i)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of global pyCart settings object
            *key*: :class:`str`
                Name of key to process
            *i*: :class:`int`
                Run index
        :Outputs:
            *rho*: :class:`float`
                Non-dimensional static density, *rho/rhoinf*
            *U*: :class:`float`
                Non-dimensional velocity, *U/ainf*
            *p*: :class:`float`
                Non-dimensional static pressure, *p/pinf*
        :Versions:
            * 2016-03-28 ``@ddalle``: First version
        """
        # Get the inputs
        p0 = self.x.GetSurfBC_TotalPressure(i, key)
        T0 = self.x.GetSurfBC_TotalTemperature(i, key)
        M  = self.x.GetSurfBC_Mach(i, key)
        # Reference pressure/temp
        pinf = self.x.GetSurfBC_RefPressure(i, key)
        Tinf = self.x.GetSurfBC_RefTemperature(i, key)
        # Freestream ratio of specific heats (Cart3D is single-species)
        gam = self.x.GetSurfBC_Gamma(i, key)
        # Calibration
        fp = self.x.GetSurfBC_PressureCalibration(i, key)
        # Calculate stagnation temperature ratio
        rT = 1 + (gam-1)/2*M*M
        # Stagnation-to-static ratios
        rr = rT ** (1/(gam-1))
        rp = rT ** (gam/(gam-1))
        # Reference values
        rho = (p0/pinf)/(T0/Tinf) / rr
        p   = fp*(p0/pinf/gam) / rp
        U   = M * np.sqrt((T0/Tinf) / rT)
        # Output
        return rho, U, p
        
    # Function to get surface BC state from *CT*
    def GetSurfCTState(self, key, i):
        """Get surface boundary state from thrust coefficient
        
        :Call:
            >>> rho, U, p = cntl.GetSurfCTState(key, i)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of global pyCart settings object
            *key*: :class:`str`
                Name of key to process
            *i*: :class:`int`
                Run index
        :Outputs:
            *rho*: :class:`float`
                Non-dimensional static density, *rho/rhoinf*
            *U*: :class:`float`
                Non-dimensional velocity, *U/ainf*
            *p*: :class:`float`
                Non-dimensional static pressure, *p/pinf*
        :Versions:
            * 2016-11-21 ``@ddalle``: First version
        """
        CT = self.x.GetSurfCT_Thrust(i, key)
        # Get the exit parameters
        M2 = self.GetSurfCT_ExitMach(key, i)
        A2 = self.GetSurfCT_ExitArea(key, i)
        # Reference values
        pinf = self.x.GetSurfCT_RefPressure(i, key)
        Tinf = self.x.GetSurfCT_RefTemperature(i, key)
        # Mach number at boundary condition (usually 1.0)
        M = self.x.GetSurfCT_Mach(i, key)
        # Ratio of specific heats
        gam = self.x.GetSurfCT_Gamma(i, key)
        # Derivative gas constants
        g2 = 0.5 * (gam-1)
        g3 = gam / (gam-1)
        # Get reference dynamic pressure
        qref = self.x.GetSurfCT_RefDynamicPressure(i, key)
        # Get reference area
        Aref = self.GetSurfCT_RefArea(key, i)
        # Get option to include pinf
        qraw = self.x.defns[key].get("RawThrust", False)
        # Calculate total pressure
        if qraw:
            # Do not account for freestream pressure
            p2 = CT*qref*Aref/A2 / (1+gam*M2*M2)
        else:
            # Account for freestream pressure
            p2 = (CT*qref*Aref + pinf*A2)/A2 / (1+gam*M2*M2)
        # Adiabatic relationship
        p0 = p2 * (1+g2*M2*M2)**g3
        # Temperature inputs
        T0 = self.x.GetSurfCT_TotalTemperature(i, key)
        # Calibration
        fp = self.x.GetSurfCT_PressureCalibration(i, key)
        # Calculate stagnation temperature ratio
        rT = 1 + (gam-1)/2*M*M
        # Stagnation-to-static ratios
        rr = rT ** (1/(gam-1))
        rp = rT ** (gam/(gam-1))
        # Reference values
        rho = (p0/pinf)/(T0/Tinf) / rr
        p   = fp*(p0/pinf/gam) / rp
        U   = M * np.sqrt((T0/Tinf) / rT)
        # Output
        return rho, U, p
        
        
    # Function to set surface BC for all components from one key
    def SetSurfBC(self, key, i, CT=False):
        """Set all SurfBCs for a particular thrust trajectory key
        
        :Call:
            >>> cntl.SetSurfBC(key, i, CT=False)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of global pyCart settings object
            *key*: :class:`str`
                Name of key to process
            *i*: :class:`int`
                Run index
            *CT*: ``True`` | {``False``}
                Inputs of thrust (``True``) or pressure (``False``)
        :Versions:
            * 2016-03-28 ``@ddalle``: First version
            * 2016-11-21 ``@ddalle``: Added *CT* input key
        """
        # Get the states
        if CT:
            # Use thrust as input variable
            rho, U, p = self.GetSurfCTState(key, i)
            # Get the components
            compIDs = self.x.GetSurfCT_CompID(i, key)
        else:
            # Use *p0* and *T0* as inputs
            rho, U, p = self.GetSurfBCState(key, i)
            # Get the components
            compIDs = self.x.GetSurfBC_CompID(i, key)
        # Ensure list
        if type(compIDs).__name__ not in ['list', 'ndarray']:
            compIDs = [compIDs]
        # Loop through the components
        for comp in compIDs:
            # Convert to list of IDs
            try:
                # Use Config.xml
                compID = self.tri.config.GetCompID(comp)
            except AttributeError:
                # Use a singleton
                compID = [comp]
            # Warn if no components
            if compID == []:
                print("  WARNING: Found no components for face '%s'" % comp)
            # Loop through the IDs
            for ci in compID:
                # Get the normal
                ni = self.tri.GetCompNormal(ci)
                # Velocity components
                u = U*ni[0]
                v = U*ni[1]
                # Check for normal
                if len(ni) > 2:
                    # Three-dimensional grid
                    w = U*ni[2]
                    # Set condition
                    self.InputCntl.SetSurfBC(ci, [rho, u, v, w, p])
                else:
                    # Two-dimensional grid
                    self.InputCntl.SetSurfBC(ci, [rho, u, v, p])
   # ]
    
   # ++++++++
   # aero.csh
   # ++++++++
   # [
    # Function re read "aero.csh" files
    def ReadAeroCsh(self):
        """Read the :file:`aero.csh` file

        :Call:
            >>> cntl.ReadAeroCsh()
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of global pyCart settings object
        :Versions:
            * 2015-10-14 ``@ddalle``: Revived from deletion
        """
        # Check for adaptation.
        if not np.any(self.opts.get_Adaptive()): return
        # Change to root safely.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # AeroCsh file name
        fname = self.opts.get_AeroCsh()
        # Check if the file exists.
        if os.path.isfile(fname):
            # Read the file.
            self.AeroCsh = AeroCsh(fname)
        else:
            # Use the template
            print("Using template for 'aero.csh' file")
            self.AeroCsh = AeroCsh(options.getCart3DTemplate('aero.csh'))
        # Go back to original location.
        os.chdir(fpwd)
    
    # Function prepare the aero.csh files
    def PrepareAeroCsh(self, i):
        """
        Write :file:`aero.csh` for run case *i* in the appropriate folder and
        with the appropriate settings.
        
        :Call:
            >>> cntl.PrepareAeroCsh(i)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of global pyCart settings object
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014-06-10 ``@ddalle``: First version
            * 2014-10-03 ``@ddalle``: Version 2.0
        """
        # Test if it's present (not required)
        try:
            self.AeroCsh
        except Exception:
            return
        # Safely go to the root folder.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get the case.
        frun = self.x.GetFullFolderNames(i)
        # Make folder if necessary.
        if not os.path.isdir(frun): self.mkdir(frun)
        # Loop through the run sequence.
        for j in range(self.opts.get_nSeq()):
            # Only write aero.csh for adaptive cases.
            if not self.opts.get_Adaptive(j): continue
            # Process options.
            self.AeroCsh.Prepare(self.opts, j)
            # Destination file name
            fout = os.path.join(frun, 'aero.%02i.csh' % j)
            # Write the input file.
            self.AeroCsh.WriteEx(fout)
        # Go back home.
        os.chdir(fpwd)
        # Done
        return None
   # ]
  # >
    
  # ========
  # PBS Jobs
  # ========
  # <
    # Write the PBS script.
    def WritePBS(self, i):
        """Write the PBS script for a given case
        
        :Call:
            >>> cntl.WritePBS(i)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014-09-30 ``@ddalle``: First version
        """
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Remember current location.
        fpwd = os.getcwd()
        # Go to the root directory.
        os.chdir(self.RootDir)
        # Make folder if necessary.
        if not os.path.isdir(frun): self.mkdir(frun)
        # Go to the folder.
        os.chdir(frun)
        # Determine number of unique PBS scripts.
        if self.opts.get_nPBS() > 1:
            # If more than one, use unique PBS script for each run.
            nPBS = self.opts.get_nSeq()
        else:
            # Otherwise use a single PBS script.
            nPBS = 1
        
        # Loop through the runs.
        for j in range(nPBS):
            # PBS script name.
            if nPBS > 1:
                # Put PBS number in file name.
                fpbs = 'run_cart3d.%02i.pbs' % j
            else:
                # Use single PBS script with plain name.
                fpbs = 'run_cart3d.pbs'
            # Initialize the PBS script.
            f = open(fpbs, 'w')
            # Write the header.
            self.WritePBSHeader(f, i, j)
            
            # Initialize options to `run_flowCart.py`
            flgs = ''

            # Get specific python version
            pyexec = self.opts.get_PythonExec(j)

            # Simply call the advanced interface.
            f.write('\n# Call the Cart3D interface.\n')
            if pyexec:
                # Use specific version
                f.write("%s -m cape.pycart run %s\n" % (pyexec, flgs))
            else:
                # Use CAPE-provided script
                f.write('run_flowCart.py' + flgs + '\n')
            
            # Close the file.
            f.close()
        # Return.
        os.chdir(fpwd)
        
  # >
    
  # ============
  # Case Options
  # ============
  # <
        
    # Extend a case
    def ExtendCase(self, i, n=1, j=None, imax=None):
        r"""Extend the number of iterations for which a case should run
        
        :Call:
            >>> cntl.ExtendCase(i, n=1, j=None, imax=None)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of pyCart control class
            *i*: :class:`int`
                Run index
            *n*: {``1``} | positive :class:`int`
                Add *n* times *steps* to the total iteration count
            *j*: {``None``} | nonnegative :class:`int`
                Apply to phase *j*, by default use the last phase
            *imax*: {``None``} | nonnegative :class:`int`
                Use *imax* as the maximum iteration count
        :Versions:
            * 2017-03-31 ``@ddalle``: Version 1.0
            * 2021-12-09 ``@ddalle``: Version 1.1; bug fixes
        """
        # Ignore cases marked PASS
        if self.x.PASS[i]:
            return
        # Read the ``case.json`` file
        rc = self.ReadCaseJSON(i)
        # Exit if none
        if rc is None:
            return
        # Process phase number (can extend middle phases)
        if j is None:
            # Use the last phase number currently in use from "case.json"
            j = rc.get_PhaseSequence(-1)
        # Get the number of steps
        NSTEPS = rc.get_it_fc(j)
        # Get the current iteration count
        ni = self.CheckCase(i)
        # Get the current cutoff for phase *j*
        if ni is None:
            # Use prescribed mas iters
            N = rc.get_PhaseIters(j)
        else:
            # Use max of prescribed target and current count
            N = max(ni, rc.get_PhaseIters(j))
        # Determine output number of steps
        if imax is None:
            # Unlimited by input; add one or more nominal runs
            N1 = N + n*NSTEPS
        else:
            # Add nominal runs but not beyond *imax*
            N1 = min(int(imax), int(N + n*NSTEPS))
        # Reset the number of steps
        rc.set_PhaseIters(N1, j)
        # Status update
        print("  Phase %i: %s --> %s" % (j, N, N1))
        # Write new options
        self.WriteCaseJSON(i, rc=rc)
            
    # Function to apply namelist settings to a case
    def ApplyCase(self, i, nPhase=None, **kw):
        """Apply settings from *cntl.opts* to an individual case
        
        This rewrites each run namelist file and the :file:`case.json` file in
        the specified directories.  It can also be used to 
        
        :Call:
            >>> cntl.ApplyCase(i, nPhase=None)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Cart3D control interface
            *i*: :class:`int`
                Case number
            *nPhase*: {``None``} | positive :class:`int`
                Last phase number (default determined by *PhaseSequence*)
        :Versions:
            * 2016-03-31 ``@ddalle``: First version
        """
        # Ignore cases marked PASS
        if self.x.PASS[i]: return
        # Case function
        self.CaseFunction(i)
        # Read ``case.json``.
        rc = self.ReadCaseJSON(i)
        # Get present options
        rco = self.opts["RunControl"]
        # Exit if none
        if rc is None: return
        # Get the number of phases in ``case.json``
        nSeqC = rc.get_nSeq()
        # Get number of phases from present options
        nSeqO = self.opts.get_nSeq()
        # Check for input
        if nPhase is None:
            # Default: inherit from pyOver.json
            nPhase = nSeqO
        else:
            # Use maximum
            nPhase = max(nSeqC, int(nPhase))
        # Present number of iterations
        nIter = rc.get_PhaseIters(nSeqC)
        # Loop through the additional phases
        for j in range(nSeqC, nPhase):
            # Append the new phase
            rc["PhaseSequence"].append(j)
            # Get iterations for this phase
            if j > nSeqO:
                # Add *nIter* iterations to last phase iter
                nj = self.opts.get_PhaseIters(j) + self.opts.get_nIter(j)
            else:
                # Use the phase break marker from master JSON file
                nj = self.opts.get_PhaseIters(j)
            # Status update
            print("  Adding phase %s (to %s iterations)" % (j, nj))
            # Set the iteration count
            nIter += nj
            rc.set_PhaseIters(nIter, j)
            # Copy other sections
            for k in rco:
                # Don't copy phase and iterations
                if k in ["PhaseIters", "PhaseSequence"]:
                    continue
                # Otherwise, overwrite
                rc[k] = rco[k]
        # Write it.
        self.WriteCaseJSON(i, rc=rc)
        # Write the conditions to a simple JSON file
        self.WriteConditionsJSON(i)
        # Rewriting phases
        print("  Writing 'input.cntl' 1 to %s" % (nPhase))
        self.PrepareInputCntl(i)
        # Rewrite 'aero.csh'` (doubt this will really work)
        print("  Writing 'aero.csh'")
        self.PrepareAeroCsh(i)
        # Write PBS scripts
        nPBS = self.opts.get_nPBS()
        print("  Writing PBS scripts 1 to %s" % (nPBS))
        self.WritePBS(i)
        
    # Function to apply settings from a specific JSON file
    def ApplyFlowCartSettings(self, **kw):
        """Apply settings from *cntl.opts* to a set of cases
        
        This rewrites the :file:`case.json` file in the specified directories.
        
        :Call:
            >>> cntl.ApplyFlowCartSettings(cons=[])
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of global pyCart settings object
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints
        :Versions:
            * 2014-12-11 ``@ddalle``: First version
        """
        # Apply filter.
        I = self.x.GetIndices(**kw)
        # Loop through cases.
        for i in I:
            # Write the JSON file.
            self.WriteCaseJSON(i)
    
  # >
    
  # ========
  # Geometry
  # ========
  # <
    # Function to create a PNG for the 3-view of each component.
    def ExplodeTri(self):
        """Create a 3-view of each named or numbered component using TecPlot
        
        This will create a folder called ``subtri/`` in the master directory for
        this *cntl* object, and it will contain a triangulation for each named
        component inf :file:`Config.xml` along with a three-view plot of each
        component created using TecPlot if possible.
        
        :Call:
            >>> cntl.ExplodeTri()
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of global pyCart settings object
        :Versions:
            * 2015-01-23 ``@ddalle``: First version
        """
        # Go to root folder safely.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Read the triangulation if necessary.
        self.ReadTri()
        # Folder name to hold subtriangulations and 3-view plots
        fdir = "subtri"
        # Create the folder if necessary.
        if not os.path.isdir(fdir): self.mkdir(fdir)
        # Go to the folder.
        os.chdir(fdir)
        # Be safe.
        try:
            # Start creating the figures and subtris.
            self.tri.TecPlotExplode()
        except:
            pass
        # Go to original location.
        os.chdir(fpwd)
    
  # >
    
  # =================
  # DataBook Updaters
  # =================
  # <
    # Function to update point sensor data book
    def UpdatePointSensor(self, **kw):
        """Update point sensor group(s) data book
        
        :Call:
            >>> cntl.UpdatePointSensor(pt=None, cons=[], **kw)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of control class containing relevant parameters
            *pt*: :class:`str`
                Optional name of point sensor group to update
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2016-01-13 ``@ddalle``: First version
        """
        # Save current location
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Read the existing data book.
        self.ReadDataBook()
        # Check for a singe point group
        pt = kw.get('pt')
        # Turn into list or default list
        if pt in [None, True]:
            # Use all components
            pts = self.opts.get_DataBookComponents()
        else:
            # Use the point given.
            pts = [pt]
        # Loop through points
        for pt in pts:
            # Make sure it's a data book point sensor group
            if self.opts.get_DataBookType(pt) != 'PointSensor': continue
            # Print name of point sensor group
            print("Updating point sensor group '%s' ..." % pt)
            # Read the point sensor group.
            self.DataBook.ReadPointSensor(pt)
            # Update it.
            self.DataBook.UpdatePointSensor(pt, I)
            # Write the updated results
            self.DataBook.PointSensors[pt].Write()
        # Return to original location.
        os.chdir(fpwd)
  # >
        
  # =========
  # Archiving
  # =========
  # <
    # Function to unarchive 'adaptXX/' folders (except for newest)
    def UntarAdapt(self, **kw):
        """Tar ``adaptNN/`` folders except for most recent one
        
        :Call:
            >>> cntl.UntarAdapt()
            >>> cntl.UntarAdapt(cons=[])
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of global pyCart settings object
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints
        :Versions:
            * 2015-04-12 ``@ddalle``: First version
        """
        # Get the format.
        opt = self.opts.get_TarAdapt()
        # Check for directive not to archive.
        if not opt or opt=="none": return
        # Save current path.
        fpwd = os.getcwd()
        # Get list of indices.
        i = self.x.GetIndices(**kw)
        # Get folder names.
        fruns = self.x.GetFullFolderNames(i)
        # Loop through folders.
        for frun in fruns:
            # Go home.
            os.chdir(self.RootDir)
            # Check for folder.
            if not os.path.isdir(frun): continue
            # Status update
            print(frun)
            # Go to the folder
            os.chdir(frun)
            # Manage the directory.
            manage.ExpandAdapt(self.opts)
        # Go back to original directory.
        os.chdir(fpwd)    
        
    # Function to archive 'adaptXX/' folders (except for newest)
    def TarAdapt(self, **kw):
        """Tar ``adaptNN/`` folders except for most recent one
        
        :Call:
            >>> cntl.TarAdapt()
            >>> cntl.TarAdapt(cons=[])
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of global pyCart settings object
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints
        :Versions:
            * 2014-11-14 ``@ddalle``: First version
            * 2014-12-10 ``@ddalle``: Added constraints
        """
        # Get the format.
        opt = self.opts.get_TarAdapt()
        # Check for directive not to archive.
        if not opt or opt=="none": return
        # Save current path.
        fpwd = os.getcwd()
        # Get list of indices.
        i = self.x.GetIndices(**kw)
        # Get folder names.
        fruns = self.x.GetFullFolderNames(i)
        # Loop through folders.
        for frun in fruns:
            # Go home.
            os.chdir(self.RootDir)
            # Check for folder.
            if not os.path.isdir(frun): continue
            # Status update
            print(frun)
            # Go to the folder
            os.chdir(frun)
            # Manage the directory.
            manage.TarAdapt(self.opts)
        # Go back to original directory.
        os.chdir(fpwd)
        
    # Function to archive 'adaptXX/' folders (except for newest)
    def TarViz(self, **kw):
        """Tar ``adaptNN/`` folders except for most recent one
        
        :Call:
            >>> cntl.TarViz()
            >>> cntl.TarViz(cons=[], **kw)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of global pyCart settings object
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints
        :Versions:
            * 2014-12-18 ``@ddalle``: First version
        """
        # Get the format.
        opt = self.opts.get_TarViz()
        fmt = self.opts.get_ArchiveFormat()
        # Check for directive not to archive.
        if not opt or opt=="none": return
        # Check for directive not to archive.
        if not fmt: return
        # Save current path.
        fpwd = os.getcwd()
        # Loop through folders.
        for i in self.x.GetIndices(**kw):
            # Get folder name.
            frun = self.x.GetFullFolderNames(i)
            # Go home.
            os.chdir(self.RootDir)
            # Check for folder.
            if not os.path.isdir(frun): continue
            # Status update
            print(frun)
            # Go to the folder
            os.chdir(frun)
            # Read the options.
            fc = case.ReadCaseJSON()
            # Check if it's unsteady.
            if not fc.get_unsteady(-1): continue
            # Manage the directory.
            manage.TarViz(self.opts)
        # Go back to original directory.
        os.chdir(fpwd)

        
    # Function to archive 'adaptXX/' folders (except for newest)
    def ArchiveCases(self, **kw):
        """Archive completed cases and clean them up if specified
        
        :Call:
            >>> cntl.ArchiveCases()
            >>> cntl.ArchiveCases(cons=[], **kw)
        :Inputs:
            *cntl*: :class:`cape.pycart.cntl.Cntl`
                Instance of global pyCart settings object
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints
        :Versions:
            * 2015-01-11 ``@ddalle``: First version
        """
        # Get the format.
        fmt = self.opts.get_ArchiveAction()
        # Check for directive not to archive.
        if not fmt or not self.opts.get_ArchiveFolder(): return
        # Save current path.
        fpwd = os.getcwd()
        # Loop through folders.
        for i in self.x.GetIndices(**kw):
            # Go to root folder.
            os.chdir(self.RootDir)
            # Get folder name.
            frun = self.x.GetFullFolderNames(i)
            # Status update
            print(frun)
            # Check if the case is ready.
            if not os.path.isdir(frun):
                print("  Folder does not exist.")
                continue
            elif self.CheckCaseStatus(i) != 'PASS':
                print("  Case is not marked PASS.")
                continue
            # Go to the folder.
            os.chdir(frun)
            # Archive.
            manage.ArchiveFolder(self.opts)
        # Go back to original directory.
        os.chdir(fpwd)
    
  # >
# class Cart3D


