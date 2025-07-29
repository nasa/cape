r"""
:mod:`cape.pyfun.cntl`: FUN3D control module
============================================

This module provides tools to quickly setup basic or complex FUN3D run
matrices and serve as an executive for pre-processing, running,
post-processing, and managing the solutions. A collection of cases
combined into a run matrix can be loaded using the following commands.

    .. code-block:: pycon

        >>> import cape.pyfun.cntl
        >>> cntl = cape.pyfun.cntl.Cntl("pyFun.json")
        >>> cntl
        <cape.pyfun.Cntl(nCase=892)>
        >>> cntl.x.GetFullFolderNames(0)
        'poweroff/m1.5a0.0b0.0'


An instance of this :class:`cape.pyfun.cntl.Cntl` class has many
attributes, which include the run matrix (``cntl.x``), the options
interface (``cntl.opts``), and optionally the data book
(``cntl.DataBook``), the appropriate input files (such as
``cntl.Namelist``), and possibly others.

    ====================   =============================================
    Attribute              Class
    ====================   =============================================
    *cntl.x*               :class:`cape.pyfun.runmatrix.RunMatrix`
    *cntl.opts*            :class:`cape.pyfun.options.Options`
    *cntl.tri*             :class:`cape.pyfun.trifile.Tri`
    *cntl.DataBook*        :class:`cape.pyfun.databook.DataBook`
    *cntl.Namelist*        :class:`cape.pyfun.namelist.Namelist`
    ====================   =============================================

Finally, the :class:`cape.pyfun.cntl.Cntl` class is subclassed from the
:class:`cape.cfdx.cntl.Cntl` class, so any methods available to the CAPE
class are also available here.

"""

# Standard library
import os
import re
import shutil

# Third-party modules
import numpy as np

# Local imports
from . import options
from . import casecntl
from . import mapbc
from . import faux
from . import databook
from . import report
from .namelist import Namelist
from .rubberdatafile import RubberData
from ..cfdx import cntl
from ..util import RangeString

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyFunFolder = os.path.split(_fname)[0]

# Adiabatic wall set
ADIABATIC_WALLBCS = (3000, 4000, 4100, 4110)

# boundary_list wall set
BLIST_WALLBCS = (
    3000, 4000, 4100, 4110,
    5051, 5052, 7011, 7012,
    7021, 7031, 7036, 7100,
    7101, 7103, 7104, 7105
)
BCS_WALL = (
    3000, 4000, 4100, 4110)

# Regular expression to parse a slice
REGEX_SLICE = re.compile(r"(?P<a>[0-9]+)([-:](?P<b>[0-9]+))?")
REGEX_IS_SLICE = re.compile(r"[0-9]+([,:-][0-9]+)*")


# Class to read input files
class Cntl(cntl.Cntl):
    r"""Class for handling global options and setup for FUN3D

    This class is intended to handle all settings used to describe a
    group of FUN3D cases.  For situations where it is not sufficiently
    customized, it can be used partially, e.g., to set up a Mach/alpha
    sweep for each single control variable setting.

    The settings are read from a JSON file, which is robust and simple
    to read, but has the disadvantage that there is no support for
    comments. Hopefully the various names are descriptive enough not to
    require explanation.

    Defaults are read from the file
    ``options/pyFun.default.json``.

    :Call:
        >>> cntl = pyFun.Cntl(fname="pyFun.json")
    :Inputs:
        *fname*: :class:`str`
            Name of pyFun input file
    :Outputs:
        *cntl*: :class:`cape.pyfun.cntl.Cntl`
            Instance of the pyFun control class
    :Data members:
        *cntl.opts*: :class:`dict`
            Dictionary of options for this case (directly from *fname*)
        *cntl.x*: :class:`pyFun.runmatrix.RunMatrix`
            Values and definitions for variables in the run matrix
        *cntl.RootDir*: :class:`str`
            Absolute path to the root directory
    :Versions:
        * 2015-10-16 ``@ddalle``: v1.0
    """
  # === Class attributes ===
    # Names
    _name = "pyfun"
    _solver = "fun3d"
    # Hooks to py{x} specific modules
    _databook_mod = databook
    _report_cls = report.Report
    # Hooks to py{x} specific classes
    _case_cls = casecntl.CaseRunner
    _opts_cls = options.Options
    # Other settings
    _fjson_default = "pyFun.json"
    _warnmode_default = cntl.DEFAULT_WARNMODE
    _zombie_files = [
        "*.out",
        "*.flow",
        "*.ugrid",
    ]

  # === Init config ===
    def init_post(self):
        r"""Do ``__init__()`` actions specific to ``pyfun``

        :Call:
            >>> cntl.init_post()
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                CAPE run matrix control instance
        :Versions:
            * 2023-05-31 ``@ddalle``: v1.0
        """
        # Read list of custom file control classes
        self.ReadNamelist()
        self.ReadMovingBodyInputFile()
        self.ReadRubberData()
        self.ReadMapBC()
        self.ReadConfig()

  # === Command-Line Interface ===
    # Baseline function
    def cli(self, *a, **kw):
        r"""Command-line interface

        :Call:
            >>> cntl.cli(*a, **kw)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
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
        # Preemptive command
        if kw.get('check'):
            # Check all
            print("---- Checking FM DataBook components ----")
            self.CheckFM(**kw)
            print("---- Checking LineLoad DataBook components ----")
            self.CheckLL(**kw)
            print("---- Checking TriqFM DataBook components ----")
            self.CheckTriqFM(**kw)
            print("---- Checking TriqPoint DataBook components ----")
            self.CheckTriqPoint(**kw)
            # Quit
            return
        elif kw.get('data', kw.get('db')):
            # Update all
            print("---- Updating FM DataBook components ----")
            self.UpdateFM(**kw)
            print("---- Updating LineLoad DataBook components ----")
            self.UpdateLL(**kw)
            print("---- Updating TriqFM DataBook components ----")
            self.UpdateTriqFM(**kw)
            print("---- Updating TriqPoint DataBook components ----")
            self.UpdateTriqPoint(**kw)
            # Output
            return
        # Call the common interface
        cmd = self.cli_cape(*a, **kw)
        # Test for a command
        if cmd is not None:
            return
        # Otherwise fall back to code-specific commands
        if kw.get('pt'):
            # Update point sensor data book
            self.UpdateTriqPoint(**kw)
        elif kw.get('checkTriqPoint') or kw.get('checkPt'):
            # Check aero databook
            self.CheckTriqPoint(**kw)
        else:
            # Submit the jobs
            self.SubmitJobs(**kw)

  # === Readers ===
    # Call special post-read DataBook functions
    def ReadDataBookPost(self):
        r"""Do ``pyfun`` specific init actions after reading DataBook

        :Call:
            >>> cntl.ReadDataBookPost()
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE run matrix control instance
        :Versions:
            * 2023-05-31 ``@ddalle``: v1.0
        """
        # Save project name
        self.DataBook.proj = self.GetProjectRootName(None)

  # === Namelist ===
    # Read the namelist
    def ReadNamelist(self, j=0, q=True):
        r"""Read the :file:`fun3d.nml` file

        :Call:
            >>> cntl.ReadNamelist(j=0, q=True)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of the pyFun control class
            *j*: :class:`int`
                Phase number
            *q*: :class:`bool`
                Whether or not to read to *Namelist*, else *Namelist0*
        :Versions:
            * 2015-10-16 ``@ddalle``: v1.0
            * 2015-12-31 ``@ddalle``: Added *Namelist0*
        """
        # Namelist file
        fnml = self.opts.get_Fun3DNamelist(j)
        # Check for empty value
        if fnml is None:
            return
        # Check for absolute path
        if not os.path.isabs(fnml):
            # Use path relative to JSON root
            fnml = os.path.join(self.RootDir, fnml)
        # Read the file
        nml = Namelist(fnml)
        # Save it.
        if q:
            # Read to main slot for modification
            self.Namelist = nml
        else:
            # Template for reading original parameters
            self.Namelist0 = nml

    # Get namelist var
    def GetNamelistVar(self, sec, key, j=0):
        r"""Get a namelist variable's value

        The JSON file overrides the value from the namelist file

        :Call:
            >>> val = cntl.GetNamelistVar(sec, key, j=0)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of global pyFun settings object
            *sec*: :class:`str`
                Name of namelist section
            *key*: :class:`str`
                Variable to read
            *j*: :class:`int`
                Run sequence index
        :Outputs:
            *val*::class:`int`|:class:`float`|:class:`str`|:class:`list`
                Value
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0
        """
        # Get the namelist value.
        nval = self.Namelist.get_opt(sec, key)
        # Check for options value.
        if nval is None:
            # No namelist file value
            return self.opts.get_namelist_var(sec, key, j)
        elif 'Fun3D' not in self.opts:
            # No namelist in options
            return nval
        elif sec not in self.opts['Fun3D']:
            # No corresponding options section
            return nval
        elif key not in self.opts['Fun3D'][sec]:
            # Value not specified in the options namelist
            return nval
        else:
            # Default to the options
            return self.opts.get_namelist_var(sec, key, j)

    # Get the project rootname
    def GetProjectRootName(self, j: int = 0) -> str:
        r"""Get the project root name

        The JSON file overrides the value from the namelist file if
        appropriate

        :Call:
            >>> name = cntl.GetProjectName(j=0)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of global pyFun settings object
            *j*: :class:`int`
                Phase number
        :Outputs:
            *name*: :class:`str`
                Project root name
        :Versions:
            * 2015-10-18 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v1.1; cleaner logic
        """
        # Read the namelist.
        self.ReadNamelist(j, False)
        # Get the namelist value
        nname = self.Namelist0.get_opt('project', 'project_rootname')
        # Get the options value
        oname = self.opts.get_project_rootname(j)
        # Check for options value
        if oname is not None:
            # Explicit JSON setting overrides
            name = oname
        elif nname is None:
            # Global default
            name = "pyfun"
        else:
            # Specified in fun3d.nml bot not pyFun.json
            name = nname
        # Check for adaptation number
        k = self.opts.get_AdaptationNumber(j)
        # Assemble project name
        if k is None:
            # No adaptation numbers
            return name
        else:
            # Append the adaptation number
            return '%s%02i' % (name, k)

    # Get the grid format
    def GetGridFormat(self, j=0):
        r"""Get the grid format

        The JSON file overrides the value from the namelist file

        :Call:
            >>> fmt = cntl.GetGridFormat(j=0)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of global pyFun settings object
            *j*: :class:`int`
                Run sequence index
        :Outputs:
            *fmt*: :class:`str`
                Project root name
        :Versions:
            * 2015-10-18 ``@ddalle``: v1.0
        """
        return self.GetNamelistVar('raw_grid', 'grid_format', j)

  # === Other Files ===
    # Read the boundary condition map
    @cntl.run_rootdir
    def ReadMapBC(self, j=0, q=True):
        r"""Read the FUN3D boundary condition map

        :Call:
            >>> cntl.ReadMapBC(q=True)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of the pyFun control class
            *q*: {``True``} | ``False``
                Whether or not to read to *MapBC*, else *MapBC0*
        :Versions:
            * 2016-03-30 ``@ddalle``: v1.0
        """
        # Read the file
        try:
            BC = mapbc.MapBC(self.opts.get_MapBCFile(j))
        except OSError:
            return
        # Save it
        if q:
            # Read to main slot.
            self.MapBC = BC
        else:
            # Template
            self.MapBC0 = BC

    # Read the ``rubber.data`` file
    @cntl.run_rootdir
    def ReadRubberData(self, j=0, q=True):
        r"""Read the :file:`rubber.data` file

        :Call:
            >>> cntl.ReadRubberData(j=0, q=True)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of the pyFun control class
            *j*: :class:`int`
                Phase number
            *q*: :class:`bool`
                Whether or not read *RubberData*, else *RubberData0*
        :Versions:
            * 2016-04-27 ``@ddalle``: v1.0
        """
        # Check if dual run
        if not self.opts.get_Dual(j):
            return
        # Get the file
        fname = self.opts.get_RubberDataFile(j)
        # Check for the file.
        if os.path.isfile(fname):
            # Read the rubber data file
            RD = RubberData(fname)
        else:
            # Use the template
            print("Using template for 'rubber.data' file")
            # Path to template file
            fname = options.getFun3DTemplate('rubber.data')
            # Read the template
            RD = RubberData(fname)
        # Save the object
        if q:
            # Read the main slot
            self.RubberData = RD
        else:
            # Template for reading original parameters
            self.RubberData0 = RD

    # Read the FAUXGeom instruction
    def ReadFAUXGeom(self):
        r"""Read any FAUXGeom input file template

        :Call:
            >>> cntl.ReadFAUXGeom()
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class
        :Versions:
            * 2017-02-23 ``@ddalle``: v1.0
        """
        # Get options
        ffaux = self.opts.get_FauxFile()
        ofaux = self.opts.get_Faux()
        # Get absolute path
        if ffaux and (not os.path.isabs(ffaux)):
            # Append root directory
            ffaux = os.path.join(self.RootDir, ffaux)
        # Read the file if appropriate
        if (ffaux is None) and (not ofaux):
            # No FAUXGeom instructions
            return
        elif ffaux and os.path.isfile(ffaux):
            # Read the file
            self.FAUXGeom = faux.FAUXGeom(ffaux)
        else:
            # Initialize an empty set of instructions
            self.FAUXGeom = faux.FAUXGeom()
        # Set instructions
        for comp in ofaux:
            # Convert *comp* to a *MapBC*
            surf = self.EvalSurfID(comp)
            # Set the geometry
            self.FAUXGeom.SetGeom(surf, ofaux[comp])

    # Read the ``moving_body.input`` file
    def ReadMovingBodyInputFile(self, j=0, q=True):
        r"""Read the :file:`moving_body.input` template

        :Call:
            >>> cntl.ReadMovingBodyInputFile(j=0, q=True)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of the pyFun control class
            *j*: :class:`int`
                Phase number
            *q*: :class:`bool`
                Whether or not to read to *Namelist*, else *Namelist0*
        :Versions:
            * 2015-10-16 ``@ddalle``: v1.0 (``ReadNamelist``)
            * 2018-10-22 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v1.1; check for file
        """
        # Namelist file
        fnml = self.opts.get_MovingBodyInputFile(j)
        # Check for empty value
        if fnml is None:
            # Empty input
            nml = None
        else:
            # Absolutize
            if not os.path.isabs(fnml):
                # Use path relative to JSON root
                fnml = os.path.join(self.RootDir, fnml)
            # Check if file is present
            if os.path.isfile(fnml):
                # Read the file
                nml = Namelist(fnml)
            else:
                # Fallback
                nml = None
        # Save it.
        if q:
            # Read to main slot for modification
            self.MovingBodyInput = nml
        else:
            # Template for reading original parameters
            self.MovingBodyInput0 = nml

    # Write FreezeSurfs
    def WriteFreezeSurfs(self, fname):
        r"""Write a ``pyfun.freeze`` file that lists surfaces to freeze

        This is about the simplest file format in history, which is
        simply a list of surface indices.

        :Call:
            >>> cntl.WriteFreezeSurfs(fname)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Control interface
            *fname*: :class:`str`
                Name of file to write
        :Versions:
            * 2017-02-23 ``@ddalle``: v1.0
        """
        # Failure tolerance
        self.ReadFreezeSurfs()
        if self.FreezeSurfs is None:
            return
        # Open the file
        f = open(fname, 'w')
        # Number of surfaces to freeze
        nfrz = len(self.FreezeSurfs)
        # Write the surfaces
        for i in range(nfrz):
            # Write the surface number
            if i + 1 == nfrz:
                # Do not write newline character
                f.write("%s" % self.FreezeSurfs[i])
            else:
                f.write("%s\n" % self.FreezeSurfs[i])
        # Close the file
        f.close()

    # Read FreezeSurfs
    def ReadFreezeSurfs(self):
        r"""Read list of surfaces to freeze

        :Call:
            >>> cntl.ReadFreezeSurfs()
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class
        :Versions:
            * 2017-02-23 ``@ddalle``: v1.0
        """
        # Check for existing list
        try:
            self.FreezeSurfs
            return
        except Exception:
            pass
        # Get the options relating to this
        ffreeze = self.opts.get_FreezeFile()
        ofreeze = self.opts.get_FreezeComponents()
        # Get absolute path
        if ffreeze and (not os.path.isabs(ffreeze)):
            # Append root directory
            ffreeze = os.path.join(self.RootDir, ffreeze)
        # Read the file if appropriate
        if (ffreeze is None) and (ofreeze is None):
            # No surfaces to read
            self.FreezeSurfs = None
        # Initialize surfaces
        surfs = []
        # Check for a file to read
        if ffreeze and os.path.isfile(ffreeze):
            # Read the file in the simplest fashion possible
            comps = open(ffreeze).read().split()
        else:
            # No list of components
            comps = []
        # Process *ofreeze*
        if ofreeze is not None:
            # Ensure list, and add it to the list from the file
            comps += list(np.array(ofreeze).flatten())
        # Loop through raw components
        for comp in comps:
            # Convert to surface
            try:
                surf = self.EvalSurfID(comp)
            except Exception:
                raise ValueError("No surface '%s' in MapBC file" % comp)
            # Check for null surface
            if surf is None:
                raise ValueError("No surface '%s' in MapBC file" % comp)
            # Check if already present
            if surf in surfs:
                continue
            # Append the surface
            surfs.append(surf)
        # Save
        self.FreezeSurfs = surfs

  # === Case ===
    # Check if cases with zero iterations are not yet setup to run
    def CheckNone(self, v=False):
        r"""Check if the current folder has the necessary files to run

        :Call:
            >>> q = cntl.CheckNone(v=False)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
            *v*: ``True`` | {``False``}
                Verbosity option
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not the case is **not** set up to run
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0
            * 2016-04-11 ``@ddalle``: Checking for AFLR3 input files,
                                      too
            * 2016-04-29 ``@ddalle``: Simpler version that handles
                                      ``Flow/``
            * 2017-02-22 ``@ddalle``: Added verbose option
        """
        # Settings file.
        if not os.path.isfile('case.json'):
            return True
        # If there's a ``Flow/`` folder, enter it
        if os.path.isdir('Flow'):
            # Dual setup
            qdual = True
            os.chdir('Flow')
        else:
            # No dual setup
            qdual = False
        # Check for history file
        for j in self.opts.get_PhaseSequence():
            # Get project name
            fproj = self.GetProjectRootName(j)
            # Check for history file
            if os.path.isfile('%s_hist.dat' % fproj):
                # Return if necessary
                if qdual:
                    os.chdir('..')
                return False
        # Namelist file
        if not os.path.isfile('fun3d.00.nml'):
            if qdual:
                os.chdir('..')
            if v:
                print("    Missing namelist file 'fun3d.00.nml'")
            return True
        # Check mesh files
        q = self.CheckMeshFiles(v=v)
        # Go back if appropriate
        if qdual:
            os.chdir('..')
        # Output
        return not q

    # Check for a failure.
    def CheckError(self, i):
        r"""Check if a case has a failure

        :Call:
            >>> q = cntl.CheckError(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                FUN3D control interface
            *i*: :class:`int`
                Run index
        :Outputs:
            *q*: :class:`bool`
                If ``True``, case has :file:`FAIL` file in it
        :Versions:
            * 2015-01-02 ``@ddalle``: v1.0
            * 2017-04-06 ``@ddalle``: Checking for
                                      ``nan_locations*.dat``
        """
        # Safely go to root.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get run name
        frun = self.x.GetFullFolderNames(i)
        # Check for the FAIL file.
        q = os.path.isfile(os.path.join(frun, 'FAIL'))
        # Check for manual marker
        q = q or self.x.ERROR[i]
        # Check for 'nan_locations*.dat'
        if not q:
            # Get list of files
            fglob = casecntl.glob.glob(
                os.path.join(frun, 'nan_locations*.dat'))
            # Check for any
            q = (len(fglob) > 0)
        # Go home.
        os.chdir(fpwd)
        # Output
        return q

  # === Mesh ===
    # Function to check if the mesh for case *i* is prepared
    def CheckMesh(self, i):
        r"""Check if the mesh for case *i* is prepared

        :Call:
            >>> q = cntl.CheckMesh(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
            *i*: :class:`int`
                Index of the case to check
        :Outputs:
            *q*: :class:`bool`
                Whether or not the mesh for case *i* is prepared
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0
        """
        # Check input
        if not type(i).__name__.startswith("int"):
            raise TypeError("Case index must be an integer")
        # Ensure case index is set
        self.opts.setx_i(i)
        # Get the group name.
        fgrp = self.x.GetGroupFolderNames(i)
        frun = self.x.GetFolderNames(i)
        # Go safely to root folder.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Check for the group folder.
        if not os.path.isdir(fgrp):
            os.chdir(fpwd)
            return False
        # Extract options
        opts = self.opts
        # Enter the group folder.
        os.chdir(fgrp)
        # Check for individual-folder mesh settings
        if not opts.get_GroupMesh():
            # Check for the case folder.
            if not os.path.isdir(frun):
                # No case folder; no mesh
                os.chdir(fpwd)
                return False
            # Enter the folder.
            os.chdir(frun)
            # Check for the Flow folder
            if self.opts.get_Dual():
                # Check for 'Flow' folder
                if not os.path.isdir('Flow'):
                    os.chdir(fpwd)
                    return False
                # Enter the folder.
                os.chdir('Flow')
        # Check for mesh files
        q = self.CheckMeshFiles()
        # Return to original folder.
        os.chdir(fpwd)
        # Output
        return q

    # Check mesh files
    def CheckMeshFiles(self, v=False):
        r"""Check for the mesh files in the present folder

        :Call:
            >>> q = cntl.CheckMeshFiles(v=False)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
            *v*: ``True`` | {``False``}
                Verbose flag
        :Outputs:
            *q*: :class:`bool`
                Whether or not the present folder has the required mesh files
        :Versions:
            * 2016-04-11 ``@ddalle``: v1.0
            * 2017-02-22 ``@ddalle``: Added verbose option
        """
        # Initialize status
        q = True
        # Get list of mesh file names
        fmesh = self.GetProcessedMeshFileNames()
        # Check for presence
        for f in fmesh:
            # Check for the file
            q = q and os.path.isfile(f)
            # Verbose option
            if v and not q:
                print("    Missing mesh file '%s'" % fmesh)
        # If running AFLR3, check for tri file
        if q and self.opts.get_aflr3():
            # Project name
            fproj = self.GetProjectRootName(0)
            # Check for mesh files
            if os.path.isfile('%s.ugrid' % fproj):
                # Have a volume mesh
                q = True
            elif os.path.isfile('%s.b8.ugrid' % fproj):
                # Have a binary volume mesh
                q = True
            elif os.path.isfile('%s.lb8.ugrid' % fproj):
                # Have a little-endian volume mesh
                q = True
            elif os.path.isfile('%s.r8.ugrid' % fproj):
                # Fortran unformatted
                q = True
            elif os.path.isfile('%s.surf' % fproj):
                # AFLR3 input file
                q = True
            elif self.opts.get_intersect():
                # Check for both required inputs
                q = os.path.isfile('%s.tri' % fproj)
                q = q and os.path.isfile('%s.c.tri' % fproj)
                # Verbose flag
                if v and not q:
                    print(
                        "    Missing TRI file for INTERSECT: '%s' or '%s'"
                        % ('%s.tri' % fproj, '%s.c.tri' % fproj))
            else:
                # No surface or mesh files
                q = False
                # Verbosity option
                if v:
                    print(
                        "    Missing mesh file '%s.{%s,%s,%s,%s,%s}'"
                        % (fproj, "ugrid", "b8.ugrid", "lb8.ugrid", "r8.ugrid",
                            "surf"))
        # Output
        return q

  # === Preparation ===
   # --- General Case ---
    # Prepare the mesh for case *i* (if necessary)
    @cntl.run_rootdir
    def PrepareMesh(self, i: int):
        r"""Prepare the mesh for case *i* if necessary

        :Call:
            >>> cntl.PrepareMesh(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0
            * 2022-04-13 ``@ddalle``: v1.1; exec_modfunction()
            * 2024-11-01 ``@ddalle``: v1.2; exfiltrate AFLR3 prep
            * 2024-11-04 ``@ddalle``: v1.3; exfiltrate *WarmStart* prep
            * 2025-03-26 ``@ddalle``: v1.4; (copy|link)_files()
        """
       # ---------
       # Case info
       # ---------
        # Ensure case index is set
        self.opts.setx_i(i)
        # Get the case name
        frun = self.x.GetFullFolderNames(i)
        # Get the name of the group
        fgrp = self.x.GetGroupFolderNames(i)
        # Create case folder
        self.make_case_folder(i)
       # ------------------
       # Folder preparation
       # ------------------
        # Check for groups with common meshes.
        if self.opts.get_GroupMesh():
            # Get the group index.
            j = self.x.GetGroupIndex(i)
            # Status update
            print("  Group name: '%s' (index %i)" % (fgrp, j))
            # Enter the group folder.
            os.chdir(fgrp)
        else:
            # Status update
            print("  Case name: '%s' (index %i)" % (frun, i))
            # Enter the case folder.
            os.chdir(frun)
            # Do we need "Adapt" and "Flow" folders?
            if self.opts.get_Dual():
                # Create folder for the primal solution
                if not os.path.isdir('Flow'):
                    os.mkdir('Flow')
                if not os.path.isdir('Adjoint'):
                    os.mkdir('Adjoint')
                # Enter
                os.chdir('Flow')
       # ----------
       # Copy files
       # ----------
        # Copy/link basic files
        self.copy_files(i)
        self.link_files(i)
        # Prepare warmstart files, if any
        warmstart = self.PrepareMeshWarmStart(i)
        # Finish if case was warm-started
        if warmstart:
            return
        # Option to linke instead of copying
        linkopt = self.opts.get_LinkMesh()
        # Get the names of the raw input files and target files
        finp = self.GetInputMeshFileNames()
        fmsh = self.GetProcessedMeshFileNames()
        # Loop through those files
        for finpj, fmshj in zip(finp, fmsh):
            # Original and final file names
            f0 = os.path.join(self.RootDir, finpj)
            f1 = fmshj
            # Copy fhe file.
            if os.path.isfile(f0) and not os.path.isfile(f1):
                if linkopt:
                    os.symlink(f0, f1)
                else:
                    shutil.copyfile(f0, f1)
       # ------------------
       # Triangulation prep
       # ------------------
        # Prepare surface triangulation for AFLR3 if appropriate
        self.PrepareMeshTri(i)

    # Prepare a case
    @cntl.run_rootdir
    def PrepareCase(self, i: int):
        r"""Prepare a case for running if it is not already prepared

        :Call:
            >>> cntl.PrepareCase(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
            *i*: :class:`int`
                Index of case to prepare/analyze
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0
        """
        # Ensure case index is set
        self.opts.setx_i(i)
        # Read MapBC file fresh
        self.ReadMapBC()
        # Get the existing status.
        n = self.CheckCase(i)
        # Quit if already prepared.
        if n is not None:
            return
        # Case function
        self.CaseFunction(i)
        # Prepare the mesh (and create folders if necessary).
        self.PrepareMesh(i)
        # Check for dual
        qdual = self.opts.get_Dual()
        # Get the run name.
        frun = self.x.GetFullFolderNames(i)
        # Create folder, just in case
        self.make_case_folder(i)
        # Enter the run directory
        os.chdir(frun)
        # Write the conditions to a simple JSON file.
        self.x.WriteConditionsJSON(i)
        # Different processes for GroupMesh and CaseMesh
        if self.opts.get_GroupMesh():
            # Required file names
            fmsh = self.GetProcessedMeshFileNames()
            # Copy the required files.
            for fname in fmsh:
                # Check for dual case
                if qdual:
                    # Link to the 'Flow/' folder
                    fto = os.path.join('Flow', fname)
                else:
                    # Link to the present folder
                    fto = fname
                # Source path
                fsrc = os.path.join(os.path.abspath('..'), fname)
                # Check for the file
                if os.path.isfile(fto):
                    os.remove(fto)
                # Create the link.
                if os.path.isfile(fsrc):
                    os.symlink(fsrc, fto)
        # Get function for setting boundary conditions, etc.
        keys = self.x.GetKeysByType('CaseFunction')
        # Get the list of functions.
        funcs = [self.x.defns[key]['Function'] for key in keys]
        # Reread namelist
        self.ReadNamelist()
        # Reread rubber.data
        if qdual:
            self.ReadRubberData()
        # Loop through the functions.
        for (key, func) in zip(keys, funcs):
            # Form args and kwargs
            a = (self, self.x[key][i])
            kw = dict(i=i)
            # Apply it
            self.exec_modfunction(func, a, kw, name="RunMatrixCaseFunction")
        # Prepare the rubber.data file
        self.PrepareRubberData(i)
        # Write the cntl.nml file(s).
        self.PrepareNamelist(i)
        # Write :file:`faux_input` if appropriate
        self.PrepareFAUXGeom(i)
        # Write list of surfaces to freeze if appropriate
        self.PrepareFreezeSurfs(i)
        # Copy gas model file
        self.PrepareTData(i)
        # Copy species thermodynamics model
        self.PrepareSpeciesThermoData(i)
        # Copy kinetic data file
        self.PrepareKineticData(i)
        # Write a JSON file with
        self.WriteCaseJSON(i)
        # Write the PBS script.
        self.WritePBS(i)

   # --- Namelist ---
    # Function to prepare "input.cntl" files
    @cntl.run_rootdir
    def PrepareNamelist(self, i: int):
        r"""Prepare and write ``fun3d.nml`` for case *i*

        :Call:
            >>> cntl.PrepareNamelist(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of FUN3D control class
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014-06-04 ``@ddalle``: v1.0
            * 2014-06-06 ``@ddalle``: v1.1; low-level grid folder funcs
            * 2014-09-30 ``@ddalle``: v1.2; single case at a time
            * 2018-04-19 ``@ddalle``: v1.3; separate flight conditions
            * 2025-04-13 ``@ddalle``: v1.4; remove PrepareAdiaba...()
        """
        # Ensure case index is set
        self.opts.setx_i(i)
        # Read namelist file
        self.ReadNamelist()
        # Set the flight conditions
        self.PrepareNamelistFlightConditions(i)
        # Get the case folder name
        frun = self.x.GetFullFolderNames(i)
        # Set up the component force & moment tracking
        self.PrepareNamelistConfig()
        # Set up boundary list
        self.PrepareNamelistBoundaryList()

        # Set the surface BCs
        for k in self.x.GetKeysByType('SurfBC'):
            # Check option for auto flow initialization
            if self.x.defns[k].get("AutoFlowInit", True):
                # Ensure the presence of the triangulation
                self.ReadTri()
            # Apply the appropriate methods
            self.SetSurfBC(k, i)
        # Set the surface BCs that use thrust as input
        for k in self.x.GetKeysByType('SurfCT'):
            # Check option for auto flow initialization
            if self.x.defns[k].get("AutoFlowInit", True):
                # Ensure the presence of the triangulation
                self.ReadTri()
            # Apply the appropriate methods
            self.SetSurfBC(k, i, CT=True)
        # File name
        if self.opts.get_Dual():
            # Write in the 'Flow/' folder
            fout = os.path.join(
                frun, 'Flow',
                '%s.mapbc' % self.GetProjectRootName(0))
        else:
            # Main folder
            fout = os.path.join(frun, '%s.mapbc' % self.GetProjectRootName(0))

        # Customize mapbc file
        self.PrepareMapBC()
        # Prepare internal boundary conditions
        self.PrepareNamelistBoundaryConditions()
        # Write the BC file
        self.MapBC.Write(fout)

        # Make folder if necessary
        self.make_case_folder(i)
        # Apply any namelist functions
        self.NamelistFunction(i)
        # Loop through input sequence
        for k, j in enumerate(self.opts.get_PhaseSequence()):
            # Set the "restart_read" property appropriately
            # This setting is overridden by *nopts* if appropriate
            if k == 0:
                # First run sequence; not restart
                self.Namelist.set_opt(
                    'code_run_control', 'restart_read', 'off')
            else:
                # Later sequence; restart
                self.Namelist.set_opt('code_run_control', 'restart_read', 'on')
            # Get the reduced namelist for sequence *j*
            nopts = self.opts.select_namelist(j)
            dopts = self.opts.select_dual_namelist(j)
            # Apply them to this namelist
            self.Namelist.apply_dict(nopts)
            # Set number of iterations
            self.Namelist.SetnIter(self.opts.get_nIter(j))
            # Ensure correct *project_rootname*
            self.Namelist.SetRootname(self.GetProjectRootName(j))
            # Check for adaptive phase
            if self.opts.get_Adaptive() and self.opts.get_AdaptPhase(j):
                # Set the project rootname of the next phase
                self.Namelist.SetAdaptRootname(self.GetProjectRootName(j+1))
                # Check for adaptive grid
                if self.opts.get_AdaptationNumber(j) > 0:
                    # Always AFLR3/stream
                    self.Namelist.set_opt('raw_grid', 'grid_format', 'aflr3')
                    self.Namelist.set_opt('raw_grid', 'data_format', 'stream')
            # Name of output file.
            if self.opts.get_Dual():
                # Write in the "Flow/" folder
                fout = os.path.join(frun, 'Flow', 'fun3d.%02i.nml' % j)
            else:
                # Write in the case folder
                fout = os.path.join(frun, 'fun3d.%02i.nml' % j)
            # Write the input file.
            self.Namelist.write(fout)
            # Check for dual phase
            if self.opts.get_Dual() and self.opts.get_DualPhase(j):
                # Apply dual options
                self.Namelist.apply_dict(dopts)
                # Write in the "Adjoint/" folder as well
                fout = os.path.join(frun, 'Flow', 'fun3d.dual.%02i.nml' % j)
                # Set restart flag appropriately
                if self.opts.get_AdaptationNumber(j) == 0:
                    # No restart read (of adjoint file)
                    self.Namelist.set_opt(
                        'code_run_control', 'restart_read', 'off')
                else:
                    # Restart read of adjoint
                    self.Namelist.set_opt(
                        'code_run_control', 'restart_read', 'on')
                    # Always AFLR3/stream
                    self.Namelist.set_opt('raw_grid', 'grid_format', 'aflr3')
                    self.Namelist.set_opt('raw_grid', 'data_format', 'stream')
                # Set the iteration count
                self.Namelist.SetnIter(self.opts.get_nIterAdjoint(j))
                # Set the adapt phase
                self.Namelist.set_opt(
                    'adapt_mechanics', 'adapt_project',
                    self.GetProjectRootName(j+1))
                # Write the adjoint namelist
                self.Namelist.write(fout)
            # Prepare moving body inputs for phase
            self.PrepareMovingBodyInputsPhase(i, j)

    @cntl.run_rootdir
    def PrepareMovingBodyInputsPhase(self, i: int, j: int):
        r"""Customize MovingBodyInputs file for case *i*

        :Call:
            >>> cntl.PrepareMovingBodyInputs(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of FUN3D control class
            *i*: :class:`int`
                Run index
            *j*: :class:`int`
                Phase index
        :Versions:
            * 2025-05-16 ``@aburkhea``: v1.0; split off PrepareNamelist
        """
        # Get the case folder name
        frun = self.x.GetFullFolderNames(i)
        # Check for valid "moving_body.input" instructions
        if not self.Namelist.get_opt("global", "moving_grid"):
            return
        # Apply "moving_body.input" parameters, if any
        self.PrepareMovingBodyPhase(j)
        # Name out oufput file
        if self.opts.get_Dual():
            # Write in the "Flow/" folder
            fout = os.path.join(
                frun, 'Flow', 'moving_body.%02i.input' % j)
        else:
            # Write in the case folder
            fout = os.path.join(frun, 'moving_body.%02i.input' % j)
        # Write the file
        self.MovingBodyInput.write(fout)

    # Apply customizations to ``.mapbc`` file
    def PrepareMapBC(self):
        r"""Customize MapBC file baed on ``"MapBC"`` section

        :Call:
            >>> cntl.PrepareMapBC()
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
        :Versions:
            * 2025-04-27 ``@ddalle``: v1.0
            * 2025-05-22 ``@ddalle``: v1.1; use MapBC IDs
        """
        # Get "MapBC" options
        bcopts = self.opts.get("MapBC", {})
        # Check for mapbc
        mapbc = getattr(self, "MapBC", None)
        # Exit if none
        if mapbc is None or bcopts is None:
            return
        # Loop through
        for surf, bc in bcopts.items():
            # Get component IDs (based on grid)
            compids = self.GetConfigBody(surf)
            # Set the BC for each
            for compid in compids:
                # Set it
                mapbc.SetBC(compid, bc)

    # Prepare moving_body.input for phase *j*
    def PrepareMovingBodyPhase(self, j: int):
        r"""Prepare ``moving_body.input`` namelist for phase *j*

        :Call:
            >>> cntl.PrepareMovingBodyPhase(j)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of FUN3D control class
            *j*: :class:`int`
                Phase index
        :Versions:
            * 2025-04-27 ``@ddalle``: v1.0
        """
        # Get namelist
        nml = self.MovingBodyInput
        # Skip if not defined
        if nml is None:
            return
        # Get moving body definitions
        nbody = self.opts.get_ConfigNBody()
        # Name of special section
        sec = "body_definitions"
        opt = "defining_bndry"
        # Loop through them
        for k in range(nbody):
            # Get definition
            comps = self.opts.get_ConfigMovingBody(k)
            # Initialize list of mapbc surf indices
            mapbcidlist = []
            # Loop through components
            for comp in comps:
                # Get list of mapbc surf indices in this component
                mapbcidlist.extend(self.GetConfigBody(comp))
            # Ensure unique list of IDs
            bodyids = np.unique(mapbcidlist)
            # Loop through each MapBC row in this "body"
            for j, i in enumerate(bodyids):
                # Set this entry
                nml.set_opt(sec, opt, i, j=(j, k))
        # Select options
        mopts = self.opts.select_moving_body_input(j)
        # Apply "moving_body.input" parameters, if any
        if mopts:
            nml.apply_dict(mopts)

    # Prepare freestream conditions
    def PrepareNamelistFlightConditions(self, i):
        r"""Set namelist flight conditions

        :Call:
            >>> cntl.PrepareNamelistFligntConditions(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of FUN3D control class
            *i*: :class:`int`
                Run index
        :Versions:
            * 2018-04-19 ``@ddalle``: v1.0
        """
        # Get equations type
        eqn_type = self.GetNamelistVar("governing_equations", "eqn_type")
        # Get temperature units
        T_units = self.GetNamelistVar(
            "reference_physical_properties", "temperature_units")
        # Default temperature units
        if T_units is None:
            T_units = "Kelvin"
        # General code for temperature units [ "K" | "R" ]
        try:
            tu = T_units[0].upper()
        except Exception:
            raise ValueError(
                "Failed to interpret temperature units [%s]" % T_units)
        # Check for generic model
        if eqn_type == "generic":
            # Set the dimensional conditions
            self.Namelist.set_opt(
                'reference_physical_properties',
                'dim_input_type', 'dimensional-SI')
            # Get properties
            a   = self.x.GetAlpha(i)
            b   = self.x.GetBeta(i)
            rho = self.x.GetDensity(i, units="kg/m^3")
            T   = self.x.GetTemperature(i, units=tu)
            V   = self.x.GetVelocity(i, units="m/s")
            # Angle of attack
            if a is not None:
                self.Namelist.SetAlpha(a)
            # Angle of sideslip
            if b is not None:
                self.Namelist.SetBeta(b)
            # Density
            if rho is not None:
                self.Namelist.SetDensity(rho)
            # Temperature
            if T is not None:
                self.Namelist.SetTemperature(T)
            # Velocity
            if V is not None:
                self.Namelist.SetVelocity(V)
        else:
            # Set the mostly nondimensional conditions
            self.Namelist.set_opt(
                'reference_physical_properties',
                'dim_input_type', 'nondimensional')
            # Get properties
            M  = self.x.GetMach(i)
            a  = self.x.GetAlpha(i)
            b  = self.x.GetBeta(i)
            Re = self.x.GetReynoldsNumber(i)
            T  = self.x.GetTemperature(i, units=tu)
            # Mach number
            if M is not None:
                self.Namelist.SetMach(M)
            # Angle of attack
            if a is not None:
                self.Namelist.SetAlpha(a)
            # Sideslip angle
            if b is not None:
                self.Namelist.SetBeta(b)
            # Reynolds number
            if Re is not None:
                self.Namelist.SetReynoldsNumber(Re)
            # Temperature
            if T is not None:
                self.Namelist.SetTemperature(T)

    # Call function to apply namelist settings for case *i*
    def NamelistFunction(self, i):
        r"""Apply a function at the end of :func:`PrepareNamelist(i)`

        This is allows the user to modify settings at a later point than
        is done using :func:`CaseFunction`

        This calls the function(s) in the global ``"NamelistFunction"``
        option from the JSON file. These functions must take *cntl* as
        an input and the case number *i*. The function(s) are usually
        from a module imported via the ``"Modules"`` option. See the
        following example:

            .. code-block:: javascript

                "Modules": ["testmod"],
                "NamelistFunction": ["testmod.nmlfunc"]

        This leads pyFun to call ``testmod.nmlfunc(cntl, i)`` near the
        end of :func:`PrepareNamelist` for each case *i* in the run
        matrix.

        :Call:
            >>> cntl.NamelistFunction(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                Overall control interface
            *i*: :class:`int`
                Case number
        :Versions:
            * 2017-06-07 ``@ddalle``: v1.0
            * 2022-04-13 ``@ddalle``: v2.0; exec_modfunction()
        :See also:
            * :func:`cape.cfdx.cntl.Cntl.CaseFunction`
            * :func:`cape.pyfun.cntl.Cntl.PrepareCase`
            * :func:`cape.pyfun.cntl.Cntl.PrepareNamelist`
        """
        # Ensure case index is set
        self.opts.setx_i(i)
        # Get input functions
        lfunc = self.opts.get("NamelistFunction", [])
        # Ensure list
        lfunc = list(np.array(lfunc).flatten())
        # Loop through functions
        for func in lfunc:
            # Form args and kwargs
            a = (self, i)
            kw = dict()
            # Apply it
            self.exec_modfunction(func, a, kw, name="NamelistFunction")

    # Set up a namelist config
    def PrepareNamelistConfig(self):
        r"""Write the lines for the force/moment output in a namelist

        :Call:
            >>> cntl.PrepareNamelistConfig()
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
        :Versions:
            * 2015-10-20 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v2.0; ``filecntl`` -> ``nmlfile``
        """
        # Get the components
        comps = self.opts.get_ConfigComponents()
        # Exit if no components
        if comps is None:
            return
        # Number
        n = len(comps)
        # Quit if nothing to do
        if n == 0:
            return
        # Option to keep template
        qkeep = self.opts.get_KeepTemplateComponents()
        # Extract namelist
        nml = self.Namelist
        # Main section name
        sec = 'component_parameters'
        # Option to keep existing template 'component_parameters'
        qkeep = self.opts.get_KeepTemplateComponents()
        # Check current list
        if qkeep:
            # Get current number of components
            n0 = nml.get_opt(sec, 'number_of_components', vdef=-1)
            # Get number if template used ``-1`` (let FUN3D decide)
            if n0 == -1:
                # Get number of "component_type"
                n0 = nml.get_opt(sec, "component_type").size
        else:
            # Start over even if something in template
            n0 = 0
        # Set the number of components
        nml.set_opt(sec, 'number_of_components', n0 + n)
        # Loop through specified components.
        for k, comp in enumerate(comps):
            # Overall index
            j = n0 + k
            # Get input definitions
            inp = self.GetConfigInput(comp)
            # Set input definitions
            inp = "" if inp is None else inp
            # Set name and inputs
            nml.set_opt(sec, 'component_name', comp, j=j)
            nml.set_opt(sec, 'component_input', inp, j=j)
            nml.set_opt(sec, 'component_count', -1, j=j)
            # Reference area
            if 'RefArea' in self.opts['Config']:
                # Get reference area.
                RefA = self.opts.get_RefArea(comp)
                # Set it
                nml.set_opt(sec, 'component_sref', RefA, j=j)
            # Moment reference center
            if 'RefPoint' in self.opts['Config']:
                # Get MRP
                RefP = self.opts.get_RefPoint(comp)
                # Set the x- and y-coordinates
                nml.set_opt(sec, 'component_xmc', RefP[0], j=j)
                nml.set_opt(sec, 'component_ymc', RefP[1], j=j)
                # Check for z-coordinate
                if len(RefP) > 2:
                    nml.set_opt(sec, 'component_zmc', RefP[2], j=j)
            # Reference length
            if 'RefLength' in self.opts['Config']:
                # Get reference length and reference span
                RefL = self.opts.get_RefLength(comp)
                RefB = self.opts.get_RefSpan(comp)
                # Set both reference lengths
                nml.set_opt(sec, 'component_cref', RefL, j=j)
                nml.set_opt(sec, 'component_bref', RefB, j=j)

    # Set boundary condition flags
    def PrepareNamelistBoundaryConditions(self):
        r"""Prepare any boundary condition flags if needed

        :Call:
            >>> cntl.PrepareNamelistBoundaryConditions()
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                FUN3D settings interface
        :Versions:
            * 2018-10-24 ``@ddalle``: v1.0
            * 2019-10-01 ``@jmeeroff``: v1.1; auto wall
            * 2022-07-13 ``@ddalle``: v1.2; "auto" flag
        """
        # Section name
        sec = "boundary_conditions"
        # Get default type
        auto_bcs = self.GetNamelistVar(sec, "adiabatic_wall")
        # Check for auto bcs
        if not auto_bcs:
            return
        # Namelist handle
        nml = self.Namelist
        # Set the wall temperature flag for adiabatic wall
        for k, bc in enumerate(self.MapBC.BCs):
            # Check for viscous wall
            if bc in BCS_WALL:
                # Get current options
                vwrf = nml.get_opt(sec, "wall_radeq_flag", k+1)
                # Escape if using wall radiative equilibrium
                if vwrf:
                    continue
                # Set wall temperature flag to .t. and temperature to -1
                nml.set_opt(sec, "wall_temp_flag", True, k+1)
                nml.set_opt(sec, "wall_temperature", -1, k+1)

    # Set adiabatic boundary condition flags
    def PrepareNamelistAdiabaticWalls(self):
        r"""Prepare any boundary condition flags if needed

        :Call:
            >>> cntl.PrepareNamelistAdiabiticWalls()
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                FUN3D settings interface
        :Versions:
            * 2021-03-22 ``@jmeeroff``: v1.0
        """
        # Namelist handle
        nml = self.Namelist
        # Save some labels
        bcs = "boundary_conditions"
        adi = "adiabatic_wall"
        wtf = "wall_temp_flag"
        wtk = "wall_temperature"
        # Get Namelist adiabatic walls entry
        if bcs in self.opts['Fun3D']:
            wallbc = self.opts['Fun3D'][bcs].pop(adi, False)
            # Check for adiabatic flag
            if wallbc:
                print("  Setting namelist options for adiabatic walls...")
                # If 'true' set all walls to adiabatic
                if isinstance(wallbc, bool):
                    for k in range(self.MapBC.n):
                        # Get the boundary type
                        BC = self.MapBC.BCs[k]
                        # Check for viscous wall
                        if BC in ADIABATIC_WALLBCS:
                            # Set the wall temperature flag for adiabatic wall
                            nml.set_opt(bcs, wtf, True, k+1)
                            nml.set_opt(bcs, wtk, -1, k+1)
                else:
                    # Ensure list
                    if type(wallbc).__name__ not in ['list', 'ndarray']:
                        wallbc = [wallbc]
                    for j in wallbc:
                        if isinstance(j, str):
                            k = self.MapBC.GetSurfIndex(
                                self.config.GetCompID(j))
                        else:
                            k = self.MapBC.GetSurfIndex(j)
                        # Get the boundary type
                        BC = self.MapBC.BCs[k]
                        # Check for viscous wall
                        if BC in ADIABATIC_WALLBCS:
                            # Set the wall temperature flag for adiabatic wall
                            nml.set_opt(bcs, wtf, True, k+1)
                            nml.set_opt(bcs, wtk, -1, k+1)
                        else:
                            raise ValueError(
                                "Trying to set non-viscous boundaries "
                                "to adiabatic, check input files...")

    # Set boundary points
    def PrepareNamelistBoundaryPoints(self):
        r"""Write the lines of the boundary point sensors in the namelist

        :Call:
            >>> cntl.PrepareNamelistBoundaryPoints()
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                FUN3D settings interface
        :Versions:
            * 2017-09-01 ``@ddalle``: v1.0
            * 2023-06-15 ``@ddalle``: v1.1; ``filecntl`` -> ``nmlfile``
        """
        # Get the boundary points
        BPG = self.opts.get_BoundaryPointGroups()
        # Check for boundary point groups
        if BPG is None:
            return
        # Number of groups
        ngrp = len(BPG)
        # Check for no points
        if ngrp == 0:
            return
        # Extract namelist
        nml = self.Namelist
        # Section name
        sec = 'sampling_parameters'
        # Existing number of geometries
        ngeom = self.GetNamelistVar(
            "sampling_parameters", "number_of_geometries")
        # If ``None``, no geometries defined
        if ngeom is None:
            ngeom = 0
        # Loop through groups
        for k in range(1, ngrp+1):
            # Get component
            grp = BPG[k-1]
            # Get the points
            PS = self.opts.get_BoundaryPoints(grp)
            # Number of points
            npt = len(PS)
            # Skip if no points
            if npt == 0:
                continue
            # Increase geometry count
            ngeom += 1
            # Set label
            nml.set_opt(sec, 'label', grp, ngeom)
            # Set the type
            nml.set_opt(sec, 'type_of_geometry', 'boundary_points', ngeom)
            # Set sampling frequency
            nml.set_opt(sec, 'sampling_frequency', -1, ngeom)
            # Set number of points
            nml.set_opt(sec, 'number_of_points', npt, ngeom)
            # Loop through points
            for j in range(1, npt+1):
                # Set point
                nml.set_opt(sec, 'points', PS[j-1], (":", ngeom, j))
        # Set number of geometries
        nml.set_opt(sec, 'number_of_geometries', ngeom)

    # Set boundary list
    def PrepareNamelistBoundaryList(self):
        r"""Write the correct boundary list in namelist

        :Call:
            >>> cntl.SetBoundaryList()
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                FUN3D settings interface
        :Versions:
            * 2021-03-18 ``@jmeeroff``: v1.0
            * 2023-05-17 ``@ddalle``: v1.1
                - check for 'boundary_output_variables' namelist
        """
        # Check for MapBC interface
        try:
            self.MapBC
        except AttributeError:
            raise AttributeError("Interface to FUN3D 'mapbc' file not found")
        # Initialize
        surf = []
        # Namelist handle
        nml = self.Namelist
        # Check if boundary list appears in json
        bov = self.opts["Fun3D"].get("boundary_output_variables", {})
        blist = bov.get('boundary_list')
        # If it exists, use these values
        if blist:
            inp = blist
        else:
            # Read from mapbc file
            # Fun3d reorders surfaces internally on 1-based system
            for k in range(self.MapBC.n):
                # Get the boundary type
                BC = self.MapBC.BCs[k]
                # Check for  wall
                if BC in BLIST_WALLBCS:
                    surf.append(k+1)
                # Sort the surface IDs to prepare RangeString
                surf.sort()
                # Convert to string
                if len(surf) > 0:
                    inp = RangeString(surf)
        # Set namelist value
        nml.set_opt('boundary_output_variables', 'boundary_list', inp)

   # --- Other Files ---
    # Prepare ``rubber.data`` file
    def PrepareRubberData(self, i):
        r"""Prepare ``rubber.data`` file if appropriate

        :Call:
            >>> cntl.PrepareRubberData(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of FUN3D control class
            *i*: :class:`int`
                Run index
        :Versions:
            * 2016-04-27 ``@ddalle``: v1.0
        """
        # Check options
        if not self.opts.get_Dual():
            return
        # Ensure case index is set
        self.opts.setx_i(i)
        # Get list of adaptive coefficients.
        coeffs = self.opts.get_FunctionalAdaptCoeffs()
        # Create list of compIDs that we've created
        comps = {}
        # Number of components in the composite adaptive function
        n = 0
        # Reread the rubber.data file
        self.ReadRubberData()
        # Save the handle
        R = self.RubberData
        # Loop through the coefficients.
        for coeff in coeffs:
            # Get the component
            comp = self.opts.get_FunctionalCoeffOpt(coeff, "compID")
            # Check if already in the list
            if comp not in comps:
                # Get the surface IDs
                comps[comp] = self.CompID2SurfID(comp)
            # Get component ID list
            surfs = comps[comp]
            # Get the option values for this coefficient
            typ = self.opts.get_FunctionalCoeffOpt(coeff, "type")
            w   = self.opts.get_FunctionalCoeffOpt(coeff, "weight")
            t   = self.opts.get_FunctionalCoeffOpt(coeff, "target")
            p   = self.opts.get_FunctionalCoeffOpt(coeff, "power")
            # Loop through the components
            for surf in surfs:
                # Increase the component count
                n += 1
                # Set the component values
                R.SetCoeffComp(1, surf, n)
                R.SetCoeffType(1, typ,  n)
                R.SetCoeffWeight(1, w, n)
                R.SetCoeffTarget(1, t, n)
                R.SetCoeffPower(1, p, n)
        # Write the file
        R.Write('rubber.data')

    # Prepare FAUXGeom file if appropriate
    @cntl.run_rootdir
    def PrepareFAUXGeom(self, i):
        r"""Prepare/edit a FAUXGeom input file for a case

        :Call:
            >>> cntl.PrepareFAUXGeom(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class
            *i*: :class:`int`
                Case index
        :Versions:
            * 2017-02-23 ``@ddalle``: v1.0
        """
        # Read options
        self.ReadFAUXGeom()
        # Check for a FAUXGeom instance
        try:
            self.FAUXGeom
        except Exception:
            # No FAUXGeom to process
            return
        # Check for more than zero planes
        if self.FAUXGeom.nSurf < 1:
            return
        # Create folder if necessary
        self.make_case_folder(i)
        # Get run folder and check if it exists
        frun = self.x.GetFullFolderNames(i)
        # Enter the run directory
        os.chdir(frun)
        # Write the file
        self.FAUXGeom.Write("faux_input")

    # Prepare list of surfaces to freeze
    @cntl.run_rootdir
    def PrepareFreezeSurfs(self, i: int):
        r"""Prepare adaption file for list of surfaces to freeze during adapts

        :Call:
            >>> cntl.PrepareFreezeSurfs(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class
            *i*: :class:`int`
                Case index
        :Versions:
            * 2017-02-23 ``@ddalle``: v1.0
        """
        # Read inputs
        self.ReadFreezeSurfs()
        # Check for something to do
        if self.FreezeSurfs is None:
            return
        # Initialize list of project root names (changes due to adapt)
        fproj = []
        # Loop through phases
        for j in self.opts.get_PhaseSequence():
            # Get the project root name for this phase
            fj = self.GetProjectRootName(j)
            # Append if not in the list
            if fj not in fproj:
                fproj.append(fj)
        # Get run folder name
        frun = self.x.GetFullFolderNames(i)
        # Create folder if necessary
        self.make_case_folder(i)
        # Enter the folder, creating if necessary
        os.chdir(frun)
        # Loop through list of project root names
        for fj in fproj:
            # Write the file
            self.WriteFreezeSurfs('%s.freeze' % fj)

    # Prepare ``tdata`` file if appropriate
    @cntl.run_rootdir
    def PrepareTData(self, i):
        r"""Prepare/edit a ``tdata`` input file for a case

        :Call:
            >>> cntl.PrepareTData(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class
            *i*: :class:`int`
                Case index
        :Versions:
            * 2018-04-19 ``@ddalle``: v1.0
        """
        # Get name of TData file (if any)
        fname = self.opts.get_TDataFile()
        # Check if it's present
        if fname is None:
            return
        # Get run folder and check if it exists
        frun = self.x.GetFullFolderNames(i)
        # Check if file is present
        if not os.path.isfile(fname):
            return
        # Create directory if necessary
        self.make_case_folder(i)
        # Destination file name
        fout = os.path.join(frun, "tdata")
        # Copy the file
        shutil.copy(fname, fout)

    # Prepare ``speciesthermodata`` file if appropriate
    @cntl.run_rootdir
    def PrepareSpeciesThermoData(self, i):
        r"""Prepare/edit a ``speciesthermodata`` input file for a case

        :Call:
            >>> cntl.PrepareSpeciesThermoData(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class
            *i*: :class:`int`
                Case index
        :Versions:
            * 2018-04-19 ``@ddalle``: v1.0
        """
        # Get name of TData file (if any)
        fname = self.opts.get_SpeciesThermoDataFile()
        # Check if it's present
        if fname is None:
            return
        # Get run folder and check if it exists
        frun = self.x.GetFullFolderNames(i)
        # Check if file is present
        if not os.path.isfile(fname):
            return
        # Create directory if necessary
        self.make_case_folder(i)
        # Destination file
        fout = os.path.join(frun, "species_thermo_data")
        # Copy the file
        shutil.copy(fname, fout)

    # Prepare ``speciesthermodata`` file if appropriate
    @cntl.run_rootdir
    def PrepareKineticData(self, i):
        r"""Prepare/edit a ``kineticdata`` input file for a case

        :Call:
            >>> cntl.PrepareKineticData(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class
            *i*: :class:`int`
                Case index
        :Versions:
            * 2018-04-19 ``@ddalle``: v1.0
        """
        # Get name of TData file (if any)
        fname = self.opts.get_KineticDataFile()
        # Check if it's present
        if fname is None:
            return
        # Get run folder and check if it exists
        frun = self.x.GetFullFolderNames(i)
        # Check if file is present
        if not os.path.isfile(fname):
            return
        # Create directory if necessary
        self.make_case_folder(i)
        # Destination file
        fout = os.path.join(frun, "kineticdata")
        # Copy the file
        shutil.copy(fname, fout)

  # === SurfCT/SurfBC ===
    # Prepare surface BC
    def SetSurfBC(self, key, i, CT=False):
        r"""Set all surface BCs and flow initialization for one key

        This uses the 7011 boundary condition and sets the values of BC
        stagnation pressure to freestream pressure and stagnation
        temperature to freestream temperature. Further, it creates a
        flow initialization volume to help with solution startup

        :Call:
            >>> cntl.SetSurfBC(key, i, CT=False)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of global pyFun settings object
            *key*: :class:`str`
                Name of SurfBC key to process
            *i*: :class:`int`
                Case index
            *CT*: ``True`` | {``False``}
                Whether this key has thrust as input (else *p0*, *T0*
                directly)
        :Versions:
            * 2016-03-29 ``@ddalle``: v1.0
            * 2016-04-13 ``@ddalle``: v1.1; SurfCT compatibility
            * 2022-06-08 ``@ddalle``: v1.2; check auto flow init
        """
        # Get options for this key
        keyopts = self.x.defns[key]
        # Auto flow initialization
        flow_init = keyopts.get("AutoFlowInit", True)
        # Get equations type
        eqn_type = self.GetNamelistVar("governing_equations", "eqn_type")
        # Get the BC inputs
        if CT:
            # Use thrust as input variable
            fp0 = self.GetSurfCTState
            typ = "SurfCT"
        else:
            # Use *p0* and *T0* directly as inputs
            fp0 = self.GetSurfBCState
            typ = "SurfBC"
        # Get the namelist
        nml = self.Namelist
        # Get the components
        compIDs = self.x.GetSurfBC_CompID(i, key, typ=typ)
        # Current number of flow initialization volumes
        n = nml.GetNFlowInitVolumes()
        # Ensure list
        if type(compIDs).__name__ not in ['list', 'ndarray']:
            compIDs = [compIDs]
        # Loop through the components
        for face in compIDs:
            # Convert to ID (if needed) and get the BC number to set
            compID = self.MapBC.GetCompID(face)
            surfID = self.MapBC.GetSurfID(compID) - 1
            # Get the BC inputs
            p0, T0 = fp0(key, i, comp=face)
            # Get the flow initialization volume state
            rho, U, a = self.GetSurfBCFlowInitState(key, i, CT=CT, comp=face)
            # Boundary condition section
            sec = 'boundary_conditions'
            # Check equation type
            if eqn_type.lower() == "generic":
                # Get BC index
                bc = keyopts.get("SurfBC", 7021)
                # Set the BC to the "rcs_jet"
                self.MapBC.SetBC(compID, bc)
                # Get gas ID number
                pID = self.x.GetSurfBC_PlenumID(i, key, typ=typ)
                # Set the BC
                nml.set_opt(sec, 'plenum_p0', p0, surfID)
                nml.set_opt(sec, 'plenum_t0', T0, surfID)
                nml.set_opt(sec, 'plenum_id', pID, surfID)
            else:
                # Get BC number
                bc = keyopts.get("SurfBC", 7011)
                # Set the BC to the correct value
                self.MapBC.SetBC(compID, bc)
                # Set the BC
                nml.set_opt(sec, 'total_pressure_ratio', p0, j=surfID)
                nml.set_opt(sec, 'total_temperature_ratio', T0, j=surfID)
            # Skip to next face if no flow initialization
            if not flow_init:
                continue
            # Get the flow initialization volume
            try:
                x1, x2, r = self.GetSurfBCVolume(key, compID)
            except Exception:
                raise ValueError(
                    "Flow init vol failed for key='%s', face='%s', compID=%i" %
                    (key, face, compID))
            # Get the surface normal
            N = self.tri.GetCompNormal(compID)
            # Velocity
            u = U * N[0]
            v = U * N[1]
            w = U * N[2]
            # Namelist section name
            sec = 'flow_initialization'
            # Set the flow initialization state.
            nml.set_opt(sec, 'rho', rho, n)
            nml.set_opt(sec, 'u', u, n)
            nml.set_opt(sec, 'v', v, n)
            nml.set_opt(sec, 'w', w, n)
            nml.set_opt(sec, 'c', a, n)
            # Initialize the flow init vol
            nml.set_opt(sec, 'type_of_volume', 'cylinder', n)
            # Set the dimensions of the volume
            nml.set_opt(sec, 'radius', r, n)
            nml.set_opt(sec, 'point1', x1, (slice(None, 3), n))
            nml.set_opt(sec, 'point2', x2, (slice(None, 3), n))
            # Increase volume number
            n += 1
        # Update number of volumes
        if flow_init:
            nml.SetNFlowInitVolumes(n)

    # Get surface BC inputs
    def GetSurfBCState(self, key, i, comp=None):
        r"""Get stagnation pressure and temperature ratios

        :Call:
            >>> p0, T0 = cntl.GetSurfBCState(key, i, comp=None)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of global pyFun settings object
            *key*: :class:`str`
                Name of key to process
            *i*: :class:`int`
                Case index
            *comp*: {``None``} | :class:`str`
                Name of component for which to get BCs
        :Outputs:
            *p0*: :class:`float`
                Ratio of BC stagnation pressure to freestream static pressure
            *T0*: :class:`float`
                Ratio of BC stagnation temperature to freestream static temp
        :Versions:
            * 2016-03-29 ``@ddalle``: v1.0
        """
        # Get equations type
        eqn_type = self.GetNamelistVar("governing_equations", "eqn_type")
        # Get the inputs
        p0 = self.x.GetSurfBC_TotalPressure(i, key, comp=comp)
        T0 = self.x.GetSurfBC_TotalTemperature(i, key, comp=comp)
        # Calibration
        ap = self.x.GetSurfBC_PressureCalibration(i, key, comp=comp)
        bp = self.x.GetSurfBC_PressureOffset(i, key, comp=comp)
        aT = self.x.GetSurfBC_TemperatureCalibration(i, key, comp=comp)
        bT = self.x.GetSurfBC_TemperatureOffset(i, key, comp=comp)
        # Reference pressure/temp
        if eqn_type == "generic":
            # Do not nondimensionalize
            pinf = 1.0
            Tinf = 1.0
        else:
            # User-specified reference conditions
            pinf = self.x.GetSurfBC_RefPressure(i, key, comp=comp)
            Tinf = self.x.GetSurfBC_RefTemperature(i, key, comp=comp)
        # Calibrated outputs
        p = (ap*p0 + bp) / pinf
        T = (aT*T0 + bT) / Tinf
        # Output
        return p, T

    # Get surface CT state inputs
    def GetSurfCTState(self, key, i, comp=None):
        r"""Get stagnation pressure and temperature ratios for *SurfCT*
        key

        :Call:
            >>> p0, T0 = cntl.GetSurfCTState(key, i, comp=None)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of global pyFun settings object
            *key*: :class:`str`
                Name of key to process
            *i*: :class:`int`
                Case index
            *comp*: {``None``} | :class:`str`
                Name of component for which to get BCs
        :Outputs:
            *p0*: :class:`float`
                Ratio of BC stagnation pressure to freestream static
                pressure
            *T0*: :class:`float`
                Ratio of BC stagnation temperature to freestream static
                temp
        :Versions:
            * 2016-04-13 ``@ddalle``: v1.0
        """
        # Get the thrust value
        CT = self.x.GetSurfCT_Thrust(i, key, comp=comp)
        # Get the exit parameters
        M2 = self.GetSurfCT_ExitMach(key, i, comp=comp)
        A2 = self.GetSurfCT_ExitArea(key, i, comp=comp)
        # Reference values
        pinf = self.x.GetSurfCT_RefPressure(i, key, comp=comp)
        Tinf = self.x.GetSurfCT_RefTemperature(i, key, comp=comp)
        # Ratio of specific heats
        gam = self.x.GetSurfCT_Gamma(i, key, comp=comp)
        # Derivative gas constants
        g2 = 0.5 * (gam-1)
        g3 = gam / (gam-1)
        # Get reference dynamic pressure
        qref = self.x.GetSurfCT_RefDynamicPressure(i, key, comp=comp)
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
        T0 = self.x.GetSurfCT_TotalTemperature(i, key, comp=comp)
        # Calibration
        ap = self.x.GetSurfCT_PressureCalibration(i, key, comp=comp)
        bp = self.x.GetSurfCT_PressureOffset(i, key, comp=comp)
        aT = self.x.GetSurfCT_TemperatureCalibration(i, key, comp=comp)
        bT = self.x.GetSurfCT_TemperatureOffset(i, key, comp=comp)
        # Output
        return (ap*p0+bp)/pinf, (aT*T0+bT)/Tinf

    # Get startup volume for a surface BC input
    def GetSurfBCVolume(self, key, compID):
        r"""Get coordinates for flow initialization box

        :Call:
            >>> x1, x2, r = cntl.GetSurfBCVolume(key, compID)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of global pyFun settings object
            *key*: :class:`str`
                Name of SurfBC key to process
            *compID*: :class:`int`
                Component ID for which to calculate flow volume
        :Outputs:
            *x1*: :class:`np.ndarray`\ [:class:`float`]
                First point of cylinder center line
            *x2*: :class:`np.ndarray`\ [:class:`float`]
                End point of cylinder center line
            *r*: :class:`float`
                Radius of cylinder
        :Versions:
            * 2016-03-29 ``@ddalle``: v1.0
        """
        # Convert to index if necessary
        compID = self.MapBC.GetCompID(compID)
        # Get the centroid
        x0 = self.tri.GetCompCentroid(compID)
        # Area and normal of the component
        A = self.tri.GetCompArea(compID)
        N = self.tri.GetCompNormal(compID)
        # Default radius
        r0 = np.sqrt(A/np.pi)
        # Process the length and radius
        r = self.x.defns[key].get("Radius", 1.1*r0)
        L = self.x.defns[key].get("Length", 4*r0)
        I = self.x.defns[key].get("Inset", 0.25*r0)
        # Set end points
        x1 = x0 - I*N
        x2 = x0 + L*N
        # Output
        return x1, x2, r

    # Get startup conditions for surface BC input
    def GetSurfBCFlowInitState(self, key, i, CT=False, comp=None):
        """Get nondimensional state for flow initialization volumes

        :Call:
            >>> rho, U, c = cntl.GetSurfBCFlowInitState(key, i,
                                                        CT=False)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of global pyFun settings object
            *key*: :class:`str`
                Name of SurfBC key to process
            *i*: :class:`int`
                Case index
            *CT*: ``True`` | {``False``}
                Whether this key has thrust as input (else *p0*,
                *T0* directly)
            *comp*: {``None``} | :class:`str`
                Name of component for which to get BCs
        :Outputs:
            *rho*: :class:`float`
                Normalized static density, *rho/rhoinf*
            *U*: :class:`float`
                Normalized velocity, *U/ainf*
            *c*: :class:`float`
                Normalized sound speed, *a/ainf*
        :Versions:
            * 2016-03-29 ``@ddalle``: v1.0
            * 2016-04-13 ``@ddalle``: Added *CT*/BC capability
        """
        # Get equations type
        eqn_type = self.GetNamelistVar("governing_equations", "eqn_type")
        # Get the boundary condition states
        if CT is True:
            # Use *SurfCT* thrust definition
            p0, T0 = self.GetSurfCTState(key, i, comp=comp)
            typ = "SurfCT"
        else:
            # Use *SurfBC* direct definitions of *p0*, *T0*
            p0, T0 = self.GetSurfBCState(key, i, comp=comp)
            typ = "SurfBC"
        # Normalize conditions for generic (real gas)
        if eqn_type.lower() == "generic":
            # Get freestream conditions
            pinf = self.x.GetPressure(i, units="Pa")
            Tinf = self.x.GetTemperature(i, units="K")
            # Normalize
            p0 = p0 / pinf
            T0 = T0 / Tinf
        # Get the Mach number and blending fraction
        M = self.x.defns[key].get('Mach', 0.2)
        f = self.x.defns[key].get('Blend', 0.9)
        # Ratio of specific heats
        gam = self.x.GetSurfBC_Gamma(i, key, typ=typ, comp=comp)
        # Calculate stagnation temperature ratio
        rT = 1 + (gam-1)/2*M*M
        # Stagnation-to-static ratios
        rr = rT ** (1/(gam-1))
        # Reference values
        rho = f*(p0)/(T0) / rr
        c   = f*np.sqrt((T0) / rT)
        U   = M * c
        # Output
        return rho, U, c

  # === Surface IDs ===
    # Get surface ID numbers
    def CompID2SurfID(self, compID):
        r"""Convert triangulation component ID to surface index

        This relies on an XML configuration file and a FUN3D ``mapbc``
        file

        :Call:
            >>> surfID = cntl.CompID2SurfID(compID)
            >>> surfID = cntl.CompID2SurfID(face)
            >>> surfID = cntl.CompID2SurfID(comps)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of FUN3D control class
            *compID*: :class:`int`
                Surface boundary ID as used in surface mesh
            *face*: :class:`str`
                Name of face
            *comps*: :class:`list` (:class:`int` | :class:`str`)
                List of component IDs or face names
        :Outputs:
            *surfID*: :class:`list`\ [:class:`int`]
                List of corresponding indices of surface in MapBC
        :Versions:
            * 2016-04-27 ``@ddalle``: v1.0
        """
        # Make sure the triangulation is present.
        try:
            self.tri
        except Exception:
            self.ReadTri()
        # Get list from tri Config
        compIDs = self.tri.config.GetCompID(compID)
        # Initialize output list
        surfID = []
        # Loop through components
        for comp in compIDs:
            # Get the surface ID
            surfID.append(self.MapBC.GetSurfID(comp))
        # Output
        return surfID

    # Convert string to MapBC surfID
    def EvalSurfID(self, comp):
        r"""Convert a component name to a MapBC surface index (1-based)

        This function also works if the input, *comp*, is an integer
        (returns the same integer) or an integer string such as ``"1"``.
        Before looking up an index by name, the function attempts to
        return ``int(comp)``.

        :Call:
            >>> surfID = cntl.EvalSurfID(comp)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class
            *comp*: :class:`str` | :class:`int`
                Component name or surface index (1-based)
        :Outputs:
            *surfID*: :class:`int`
                Surface index (1-based) according to *cntl.MapBC*
        :Versions:
            * 2017-02-23 ``@ddalle``: v1.0
        """
        # Try to convert input to an integer directly
        try:
            return int(comp)
        except Exception:
            pass
        # Check for MapBC interface
        try:
            self.MapBC
        except AttributeError:
            raise AttributeError("Interface to FUN3D 'mapbc' file not found")
        # Read from MapBC
        return self.MapBC.GetSurfID(comp)

    # Get string describing which components are in config
    def GetConfigBody(self, comp: str, warn: bool = False) -> list:
        r"""Convert face name to list of MapBC indices

        Determine which component indices are in a named component based
        on the MapBC file, which is always numbered 1,2,...,N.  Output
        the format as a nice string, such as ``"4-10,13,15-18"``.

        If possible, this is read from the ``"Inputs"`` subsection of
        the ``"Config"`` section of the master JSON file.  Otherwise,
        it is read from the ``"mapbc"`` and configuration files.

        :Call:
            >>> cids = cntl.GetConfigInput(comp, warn=False)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
            *comp*: :class:`str`
                Name of component to process
            *warn*: ``True`` | {``False``}
                Whether or not to print warnings if not raising errors
        :Outputs:
            *cids*: :class:`list`\ [:class:`str`]
                List of MapBC inds (1-based) in *comp*
        :Versions:
            * 2025-05-16 ``@ddalle``: v1.0 (from GetConfigInput())
        """
        # Initialize
        surf = []
        # Get names of all child components, including *comp*
        family = self.config.GetFamily(comp)
        # Loop through components
        for face in family:
            # Check if present
            if face not in self.MapBC.Names:
                continue
            # Get the surf from MapBC
            surfID = self.MapBC.GetSurfIndex(face, check=True, warn=False) + 1
            # If one was found, append it
            if surfID is not None:
                surf.append(surfID)
        # Check for empty
        if len(surf) == 0:
            print(f"     Component '{comp}' has no matches in mapbc file")
            return []
        # Sort the surface IDs to prepare RangeString
        surf.sort()
        return surf

    # Get string describing which components are in config
    def GetConfigInput(self, comp: str, warn: bool = False):
        r"""Convert face name to list of MapBC indices

        Determine which component indices are in a named component based
        on the MapBC file, which is always numbered 1,2,...,N.  Output
        the format as a nice string, such as ``"4-10,13,15-18"``.

        If possible, this is read from the ``"Inputs"`` subsection of
        the ``"Config"`` section of the master JSON file.  Otherwise,
        it is read from the ``"mapbc"`` and configuration files.

        :Call:
            >>> cntl.GetConfigInput(comp, warn=False)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
            *comp*: :class:`str`
                Name of component to process
            *warn*: ``True`` | {``False``}
                Whether or not to print warnings if not raising errors
        :Outputs:
            *inp*: :class:`str`
                String describing list of integers included
        :Versions:
            * 2016-10-21 ``@ddalle``: v1.0
            * 2025-03-13 ``@ddalle``: v2.0; use config.GetFamily()
        """
        # Get input definitions.
        inp = self.opts.get_ConfigInput(comp)
        # Determine from MapBC probably
        if inp is not None:
            return inp
        # Otherwise, read from the MapBC interface
        try:
            self.MapBC
            self.config
        except Exception:
            return
        # Determine entries from MapBC
        surf = self.GetConfigBody(comp, warn=warn)
        if surf is None or len(surf) == 0:
            return ""
        # Convert to string
        inp = RangeString(surf)
        return inp

  # === Case Modification ===
    # Function to apply namelist settings to a case
    def ApplyCase(self, i: int, nPhase=None, **kw):
        r"""Apply settings from *cntl.opts* to an individual case

        This rewrites each run namelist file and the :file:`case.json`
        file in the specified directories.

        :Call:
            >>> cntl.ApplyCase(i, nPhase=None)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                FUN3D control interface
            *i*: :class:`int`
                Case number
            *nPhase*: {``None``} | positive :class:`int`
                Last phase number (default determined by *PhaseSequence*)
        :Versions:
            * 2016-03-31 ``@ddalle``: v1.0
        """
        # Ignore cases marked PASS
        if self.x.PASS[i] or self.x.ERROR[i]:
            return
        # Case function
        self.CaseFunction(i)
        # Read ``case.json``.
        rc = self.read_case_json(i)
        # Get present options
        rco = self.opts["RunControl"]
        # Exit if none
        if rc is None:
            return
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
        # Get nominal phase breaks
        PhaseIters = self.GetPhaseBreaks()
        # Loop through the additional phases
        for j in range(nSeqC, nPhase):
            # Append the new phase
            rc["PhaseSequence"].append(j)
            # Get iterations for this phase
            if j >= nSeqO:
                # Add *nIter* iterations to last phase iter
                nj = self.opts.get_nIter(j)
            else:
                # Process number of *additional* iterations expected
                nj = PhaseIters[j] - PhaseIters[j-1]
            # Set the iteration count
            nIter += nj
            rc.set_PhaseIters(nIter, j)
            # Status update
            print("  Adding phase %s (to %s iterations)" % (j, nIter))
        # Copy other sections
        for k in rco:
            # Don't copy phase and iterations
            if k in ["PhaseIters", "PhaseSequence"]:
                continue
            # Otherwise, overwrite
            rc[k] = rco[k]
        # Write it
        self.WriteCaseJSON(i, rc=rc)
        # Write the conditions to a simple JSON file
        self.WriteConditionsJSON(i)
        # (Re)Prepare mesh in case needed
        print("  Checking mesh preparations")
        self.PrepareMesh(i)
        # Rewriting phases
        print("  Writing input namelists 0 to %s" % (nPhase-1))
        self.PrepareNamelist(i)
        # Write PBS scripts
        nPBS = self.opts.get_nPBS()
        print("  Writing PBS scripts 0 to %s" % (nPBS-1))
        self.WritePBS(i)

  # === Case Interface ===
    # Read a namelist from a case folder
    def ReadCaseNamelist(self, i: int, j=None):
        r"""Read namelist from case *i*, phase *j* if possible

        :Call:
            >>> nml = cntl.ReadCaseNamelist(i, rc=None, j=None)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of FUN3D control class
            *i*: :class:`int`
                Run index
            *j*: {``None``} | nonnegative :class:`int`
                Phase number
        :Outputs:
            *nml*: ``None`` | :class:`pyOver.overnmlfile.OverNamelist`
                Namelist interface is possible
        :Versions:
            * 2016-12-12 ``@ddalle``: v1.0
            * 2023-09-18 ``@ddalle``: v2.0; use ``CaseRunner``
        """
        # Read case runner
        runner = self.ReadCaseRunner(i)
        # If no case, abort
        if runner is None:
            return
        # Read the namelist
        return runner.read_namelist(j=j)

