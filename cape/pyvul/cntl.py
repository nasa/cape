r"""
:mod:`cape.pyvul.cntl`: VULCAN-CFD control module
===================================================

This module provides tools to setup and run VULCAN-CFD in CAPE.

    .. code-block:: pycon

        >>> import cape.pyvul.cntl
        >>> cntl = cape.pyvul.cntl.Cntl("pyVul.json")
        >>> cntl
        <cape.pyvul.Cntl(nCase=892)>
        >>> cntl.x.GetFullFolderNames(0)
        'poweroff/m1.5a0.0b0.0'


An instance of this :class:`cape.pyvul.cntl.Cntl` class has many
attributes, which include the run matrix (``cntl.x``), the options
interface (``cntl.opts``), and the appropriate input files (such as
``cntl.VulcanInp``), and possibly others.

    ====================   =============================================
    Attribute              Class
    ====================   =============================================
    *cntl.x*               :class:`cape.pyvul.runmatrix.RunMatrix`
    *cntl.opts*            :class:`cape.pyvul.options.Options`
    *cntl.tri*             :class:`cape.trifile.Tri`
    *cntl.inp*             :class:`cape.pyvul.inpfile.VulcanInpFile`
    ====================   =============================================

Finally, the :class:`cape.pyfun.cntl.Cntl` class is subclassed from the
:class:`cape.cfdx.cntl.Cntl` class, so any methods available to the CAPE
class are also available here.

"""

# Standard library
import os
import re
import shutil
from math import cos, radians, sin, sqrt

# Third-party modules
import numpy as np

# Local imports
from . import options
from . import casecntl
from .inpfile import VulcanInpFile
from ..cfdx import cntl

# Regular expression to parse a slice
REGEX_SLICE = re.compile(r"(?P<a>[0-9]+)([-:](?P<b>[0-9]+))?")
REGEX_IS_SLICE = re.compile(r"[0-9]+([,:-][0-9]+)*")


# Class to read input files
class Cntl(cntl.Cntl):
    r"""Class for handling global options and setup for VULCAN-CFD

    This class is intended to handle all settings used to describe a
    group of VULCAN cases.  For situations where it is not sufficiently
    customized, it can be used partially, e.g., to set up a Mach/alpha
    sweep for each single control variable setting.

    The settings are read from a JSON file, which is robust and simple
    to read, but has the disadvantage that there is no support for
    comments. Hopefully the various names are descriptive enough not to
    require explanation.

    :Call:
        >>> cntl = Cntl(fname="pyVul.json")
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
    """
  # === Class attributes ===
    # Names
    _name = "pyvul"
    _solver = "vulcan"
    # Hooks to py{x} specific modules
    # Hooks to py{x} specific classes
    _case_cls = casecntl.CaseRunner
    _opts_cls = options.Options
    # Other settings
    _fjson_default = "pyVul.json"
    _warnmode_default = cntl.DEFAULT_WARNMODE
    _file_opts = [
        ".VulcanInpFile",
        ".Mesh.MapBCFile",
    ]
    _zombie_files = [
        "*.out",
        "*.flow",
        "*.ugrid",
    ]

  # === Init config ===
    def init_post(self):
        self.ReadVulcanInpFile()

  # === Main Input File ===
    # Read the namelist
    def ReadVulcanInpFile(self, j: int = 0, q: bool = True):
        r"""Read the :file:`fun3d.nml` file

        :Call:
            >>> cntl.ReadVulcanInpFile(j=0, q=True)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of the pyFun control class
            *j*: :class:`int`
                Phase number
            *q*: :class:`bool`
                Whether or not to read to *Namelist*, else *Namelist0*
        """
        # Namelist file
        finp = self.opts.get_VulcanInpFile(j)
        # Check for empty value
        if finp is None:
            return
        # Check for absolute path
        if not os.path.isabs(finp):
            # Use path relative to JSON root
            finp = os.path.join(self.RootDir, finp)
        # Read the file
        inp = VulcanInpFile(finp)
        # Save it.
        if q:
            # Read to main slot for modification
            self.inp = inp
        else:
            # Template for reading original parameters
            self.inp0 = inp

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

  # === Case ===
    # Check if cases with zero iterations are not yet setup to run
    def CheckNone(self, v: bool = False) -> bool:
        return False

    # Check for a failure
    @cntl.run_rootdir
    def CheckError(self, i: int) -> bool:
        r"""Check if a case has a failure

        :Call:
            >>> q = cntl.CheckError(i)
        :Inputs:
            *cntl*: :class:`cape.pyvul.cntl.Cntl`
                VULCAN control interface
            *i*: :class:`int`
                Run index
        :Outputs:
            *q*: :class:`bool`
                If ``True``, case has :file:`FAIL` file in it
        """
        # Get run name
        frun = self.x.GetFullFolderNames(i)
        # Check for the FAIL file.
        q = os.path.isfile(os.path.join(frun, 'FAIL'))
        # Check for manual marker
        q = q or self.x.ERROR[i]
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
        self.ReadVulcanInpFile()
        # Loop through the functions.
        for (key, func) in zip(keys, funcs):
            # Form args and kwargs
            a = (self, self.x[key][i])
            kw = dict(i=i)
            # Apply it
            self.exec_modfunction(func, a, kw, name="RunMatrixCaseFunction")
        # Write the cntl.nml file(s).
        self.PrepareVulcanInp(i)
        # Write a JSON file with
        self.WriteCaseJSON(i)
        # Write the PBS script.
        self.WritePBS(i)

   # --- Namelist ---
    # Function to prepare "input.cntl" files
    @cntl.run_rootdir
    def PrepareVulcanInp(self, i: int):
        r"""Prepare and write ``vulcan.inp`` for case *i*

        :Call:
            >>> cntl.PrepareNamelist(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of FUN3D control class
            *i*: :class:`int`
                Run index
        """
        # Ensure case index is set
        self.opts.setx_i(i)
        # Read namelist file
        self.ReadVulcanInpFile()
        pass
        # Set the flight conditions
        self.PrepareVulcanInpFlightConditions(i)
        # Get the case folder name
        frun = self.x.GetFullFolderNames(i)
        # Set up the component force & moment tracking
        self.PrepareVulcanInpConfig()
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

    # Prepare freestream conditions
    def PrepareVulcanInpFlightConditions(self, i: int):
        r"""Set VULCAN input file flight conditions

        :Call:
            >>> cntl.PrepareVulcanInpFlightConditions(i)
        :Inputs:
            *cntl*: :class:`cape.pyvul.cntl.Cntl`
                Instance of CAPE control class
            *i*: :class:`int`
                Run index
        :Versions:
            * 2026-09-25 ``@ddalle``: v1.0
        """
        # Ensure case index is set
        self.opts.setx_i(i)
        # Check for input file
        inp = getattr(self, "inp", None)
        if inp is None:
            return
        # Get properties
        M = self.x.GetMach(i)
        a = self.x.GetAlpha(i)
        b = self.x.GetBeta(i)
        p = self.x.GetPressure(i, units="Pa")
        T = self.x.GetTemperature(i, units="K")
        # Mach number
        if M is not None:
            inp.set_mach(M)
        # Angle of attack
        if a is not None:
            inp.set_alpha(a)
        # Angle of sideslip
        if b is not None:
            inp.set_beta(b)
        # Static pressure
        if p is not None:
            inp.set_pressure(p)
        # Static temperature
        if T is not None:
            inp.set_temperature(T)
        # Apply the freestream state to the farfield BC groups
        self.PrepareVulcanInpFarfield(i)

    # Apply freestream state to farfield boundary conditions
    def PrepareVulcanInpFarfield(self, i: int):
        r"""Set the constant-state data of farfield ``FIX_IN`` groups

        The groups to update come from the ``"FarfieldComponents"``
        option of the ``"Config"`` section; if that list is empty,
        every ``FIX_IN`` group is used. The velocity components follow
        the VULCAN body axes (``x`` forward, ``y`` right wing, ``z``
        down) with
        ::

            u = V cos(alpha) cos(beta)
            v = V sin(beta)
            w = V sin(alpha)

        :Call:
            >>> cntl.PrepareVulcanInpFarfield(i)
        :Inputs:
            *cntl*: :class:`cape.pyvul.cntl.Cntl`
                Instance of CAPE control class
            *i*: :class:`int`
                Run index
        :Versions:
            * 2026-09-25 ``@ddalle``: v1.0
        """
        # Ensure case index is set
        self.opts.setx_i(i)
        # Check for input file & BC groups
        inp = getattr(self, "inp", None)
        if inp is None:
            return
        bcg = inp.bcgroups
        if len(bcg) == 0:
            return
        # Get the list of farfield components
        comps = self.opts.get_FarfieldComponents()
        comps = [] if comps is None else list(comps)
        # Fall back to all ``FIX_IN`` groups
        if len(comps) == 0:
            comps = bcg.find_fixin()
        # Quit if there is nothing to update
        if len(comps) == 0:
            return
        # Get freestream properties
        M = self.x.GetMach(i)
        a = self.x.GetAlpha(i)
        b = self.x.GetBeta(i)
        p = self.x.GetPressure(i, units="Pa")
        T = self.x.GetTemperature(i, units="K")
        rho = self.x.GetDensity(i, units="kg/m^3")
        V = self.x.GetVelocity(i, units="m/s")
        # Gas properties from the input file
        Rg = inp.get("GAS CONSTANT")
        if Rg is None:
            Rg = 287.052
        gam = inp.get("GAMMA")
        if gam is None:
            gam = 1.4
        # Derive the density if necessary
        if rho is None and (p is not None) and (T is not None):
            rho = p / (Rg * T)
        # Derive the velocity if necessary
        if V is None and (M is not None) and (T is not None):
            V = M * sqrt(gam * Rg * T)
        # Velocity components along VULCAN body axes
        uu = vv = ww = None
        if V is not None:
            aa = 0.0 if a is None else a
            bb = 0.0 if b is None else b
            uu = V * cos(radians(aa)) * cos(radians(bb))
            vv = V * sin(radians(bb))
            ww = V * sin(radians(aa))
        # Freestream state values
        state = {
            "density": rho,
            "uvel": uu,
            "vvel": vv,
            "wvel": ww,
            "temperature": T,
        }
        # Loop through the requested components
        for comp in comps:
            # Check for the group
            if comp not in bcg:
                raise ValueError(
                    "Farfield component '%s' not in 'BC GROUPS' of '%s'"
                    % (comp, inp.fname))
            # Skip groups without a constant-state line
            if not bcg[comp].get_state():
                print(
                    "  Warning: BC group '%s' has no constant-state"
                    " data line; skipping" % comp)
                continue
            # Apply the freestream state
            bcg[comp].set_state(state)

    # Call function to apply namelist settings for case *i*
    def VulcanInpFunction(self, i: int):
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
        lfunc = self.opts.get("VulcanInpFunction", [])
        # Ensure list
        lfunc = list(np.array(lfunc).flatten())
        # Loop through functions
        for func in lfunc:
            # Form args and kwargs
            a = (self, i)
            kw = dict()
            # Apply it
            self.exec_modfunction(func, a, kw, name="VulcanInpFunction")

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
        if warn and (len(surf) == 0):
            print(f"     Component '{comp}' has no matches in mapbc file")
            return []
        # Sort the surface IDs to prepare RangeString
        surf.sort()
        return surf

  # === Case Modification ===
    # Get case-specific number of iterations for a phase run
    def get_phase_niter(self, i: int, j: int) -> int:
        pass

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
        self.PrepareVulcanInp(i)
        # Write PBS scripts
        nPBS = self.opts.get_nPBS()
        print("  Writing PBS scripts 0 to %s" % (nPBS-1))
        self.WritePBS(i)

  # === Case Interface ===
    # Read a namelist from a case folder
    def ReadCaseVulcanInp(self, i: int, j: int | None = None) -> VulcanInpFile:
        r"""Read inputs from case *i*, phase *j* if possible

        :Call:
            >>> inp = cntl.ReadCaseVulcanInp(i, rc=None, j=None)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of CAPE control class
            *i*: :class:`int`
                Run index
            *j*: {``None``} | nonnegative :class:`int`
                Phase number
        :Outputs:
            *inp*: ``None`` | :class:`VulcanInpFile`
                Namelist interface is possible
        """
        # Read case runner
        runner = self.ReadCaseRunner(i)
        # If no case, abort
        if runner is None:
            return
        # Read the namelist
        return runner.read_vulcan_inp(j=j)

