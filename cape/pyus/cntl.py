#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pyus.us3d`: US3D control module
============================================

This module provides tools to quickly setup basic or complex US3D run matrices
and serve as an executive for pre-processing, running, post-processing, and
managing the solutions. A collection of cases combined into a run matrix can be
loaded using the following commands.

    .. code-block:: pycon

        >>> import cape.pyus.us3d
        >>> cntl = cape.pyus.us3d.US3D("pyUS.json")
        >>> cntl
        <cape.pyus.US3D(nCase=892)>
        >>> cntl.x.GetFullFolderNames(0)
        'poweroff/m1.5a0.0b0.0'


An instance of this :class:`cape.pyus.cntl.Cntl` class has many
attributes, which include the run matrix (``cntl.x``), the options
interface (``cntl.opts``), and optionally the data book
(``cntl.DataBook``), the appropriate input files (such as
``cntl.InputInp``), and possibly others.

    ====================   =============================================
    Attribute              Class
    ====================   =============================================
    *cntl.x*               :class:`cape.pyus.RunMatrix`
    *cntl.opts*            :class:`cape.pyus.options.Options`
    *cntl.DataBook*        :class:`cape.pyus.dataBook.DataBook`
    *cntl.InputInp*        :class:`cape.pyus.inputInp.InputInp`
    ====================   =============================================

Finally, the :class:`cape.pyus.cntl.Cntl` class is subclassed from the
:class:`cape.cntl.Cntl` class, so any methods available to the CAPE class are
also available here.

"""

# System modules
import os
import shutil

# Third-party modules
import numpy as np

# CAPE classes and specific imports
from .. import cntl
from . import options
from . import case
from .runmatrix import RunMatrix

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyUSFolder = os.path.split(_fname)[0]


# Class to read input files
class Cntl(cntl.Cntl):
    """
    Class for handling global options and setup for US3D.

    This class is intended to handle all settings used to describe a group
    of US3D cases.  For situations where it is not sufficiently
    customized, it can be used partially, e.g., to set up a Mach/alpha sweep
    for each single control variable setting.

    The settings are read from a JSON file, which is robust and simple to
    read.  The settings can be extended using Python modules if desired.

    Defaults are read from the file ``$CAPE/settings/pyUS.default.json``.

    :Call:
        >>> cntl = cape.pyus.Cntl(fname="pyUS.json")
    :Inputs:
        *fname*: {``"pyUS.json"``} | :class:`str`
            Name of pyUS JSON control file
    :Outputs:
        *cntl*: :class:`cape.pyfun.cntl.Cntl`
            Instance of control class
    :Data members:
        *cntl.opts*: :class:`dict`
            Dictionary of options for this case (directly from *fname*)
        *cntl.x*: :class:`cape.pyus.runmatrix.RunMatrix`
            Values and definitions for variables in the run matrix
        *cntl.RootDir*: :class:`str`
            Absolute path to the root directory
    :Versions:
        * 2019-06-04 ``@ddalle``: Started
    """
  # ================
  # Class attributes
  # ================
  # <
    # Case module
    _case_mod = case
    # Options class
    _opts_cls = options.Options
  # >

  # ======
  # Config
  # ======
  # <
    # Initialization method
    def __init__(self, fname="pyUS.json"):
        """Initialization method for :mod:`cape.cntl.Cntl`"""
        # Check if file exists
        if not os.path.isfile(fname):
            # Raise error but suppress traceback
            os.sys.tracebacklimit = 0
            raise ValueError("No pyUS control file '%s' found" % fname)

        # Get the real path
        fjson = os.path.realpath(fname)
        # Save it
        self.fname = fjson

        # Read settings
        self.read_options(fname)

        #Save the current directory as the root
        self.RootDir = os.getcwd()

        # Import modules
        self.modules = {}
        self.ImportModules()

        # Process the trajectory.
        self.x = RunMatrix(**self.opts['RunMatrix'])

        # Job list
        self.jobs = {}

        # Read the input file template(s)
        self.ReadInputInp()

        # Read the configuration
        #self.ReadConfig()

        # Set umask
        os.umask(self.opts.get_umask())

        # Run any initialization functions
        self.InitFunction()

    # Output representation
    def __repr__(self):
        """Output representation for the class."""
        # Display basic information from all three areas.
        return '<cape.pyus.US3D("%s", nCase=%i)>' % (
            os.path.split(self.fname)[1],
            self.x.nCase)
  # >

  # ============
  # Preparation
  # ============
  # <
   # -------------
   # General Case
   # -------------
   # [

    # Prepare a case
    def PrepareCase(self, i):
        r"""Prepare a case for running if it is not already prepared

        :Call:
            >>> cntl.PrepareCase(i)
        :Inputs:
            *fun3d*: :class:`cape.pyus.cntl.Cntl`
                Instance of control class
            *i*: :class:`int`
                Index of case to prepare/analyze
        :Versions:
            * 2015-10-19 ``@ddalle``: Version 1.0 (pyfun)
            * 2020-04-13 ``@ddalle``: Version 1.0
            * 2022-04-13 ``@ddalle``: Version 1.1; exec_modfunction()
        """
        # Get the existing status
        n = self.CheckCase(i)
        # Quit if already prepared
        if n is not None:
            return
        # Go to root folder safely
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Case function
        self.CaseFunction(i)
        # Prepare the mesh (and create folders if necessary).
        self.PrepareMesh(i)
        # Get the run name.
        frun = self.x.GetFullFolderNames(i)
        # Create run directory if necessary
        if not os.path.isdir(frun):
            self.mkdir(frun)
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
        self.ReadInputInp()
        # Loop through the functions.
        for (key, func) in zip(keys, funcs):
            # Form args and kwargs
            a = (self, self.x[key][i])
            kw = dict(i=i)
            # Apply it
            self.exec_modfunction(func, a, kw, name="RunMatrixCaseFunction")
        # Write the "input.inp" file(s)
        self.PrepareInputInp(i)
        # Write a JSON file with
        self.WriteCaseJSON(i)
        # Write the PBS script.
        self.WritePBS(i)
        # Return to original location
        os.chdir(fpwd)

    # Prepare the mesh for case *i* (if necessary)
    def PrepareMesh(self, i):
        r"""Prepare the mesh for case *i* if necessary

        :Call:
            >>> cntl.PrepareMesh(i)
        :Inputs:
            *cntl*: :class:`cape.pyus.cntl.Cntl`
                Instance of control class
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
            * 2020-04-13 ``@ddalle``: Forked from :mod:`cape.pyfun`
        """
       # ---  Case info ---
        # Check if the mesh is already prepared
        qmsh = self.CheckMesh(i)
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Get the name of the group.
        fgrp = self.x.GetGroupFolderNames(i)
       # --- Folder preparation ---
        # Remember current location.
        fpwd = os.getcwd()
        # Go to root folder.
        os.chdir(self.RootDir)
        # Check for the group folder and make it if necessary.
        if not os.path.isdir(fgrp):
            self.mkdir(fgrp)
        # Check for groups with common meshes.
        if self.opts.get_GroupMesh():
            # Get the group index.
            j = self.x.GetGroupIndex(i)
            # Status update
            print("  Group name: '%s' (index %i)" % (fgrp, j))
            # Enter the group folder.
            os.chdir(fgrp)
        else:
            # Check if the fun folder exists.
            if not os.path.isdir(frun):
                self.mkdir(frun)
            # Status update
            print("  Case name: '%s' (index %i)" % (frun, i))
            # Enter the case folder.
            os.chdir(frun)
       # --- Copy files ---
        # Get the names of the raw input files and target files
        finp = self.GetInputMeshFileNames()
        fmsh = self.GetProcessedMeshFileNames()
        # Project name
        fproj = self.GetProjectRootName(0)
        # Loop through those files
        for j, fi in enumerate(finp):
            # Source file
            if os.path.isabs(fi):
                # Already absolute
                f0 = fi
            else:
                # Relative to root directory
                f0 = os.path.join(self.RootDir, fi)
            # Output file name
            f1 = fmsh[j]
            # Copy fhe file.
            if os.path.isfile(f0) and not os.path.isfile(f1):
                shutil.copyfile(f0, f1)
       # --- Cleanup ---
        # Return to original folder
        os.chdir(fpwd)
   # ]

   # ------------
   # "input.inp"
   # ------------
   # [
    # Function to prepare "input.cntl" files
    def PrepareInputInp(self, i):
        r"""Prep and write ``input.inp`` with case-specific settings

        :Call:
            >>> cntl.PrepareNamelist(i)
        :Inputs:
            *cntl*: :class:`cape.pyus.cntl.Cntl`
                Instance of control class
            *i*: :class:`int`
                Run index
        :Versions:
            * 2020-04-13 ``@ddalle``: First version
        """
        # Read namelist file
        self.ReadInputInp()
        # Handle
        inp = self.InputInp
        # Go safely to root folder.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Set the flight conditions
        self.PrepareInputInpFlightConditions(i)

        # Get the case folder name
        frun = self.x.GetFullFolderNames(i)
        # Prepare other special blocks...


        # Make folder if necessary
        if not os.path.isdir(frun):
            self.mkdir(frun)
        # Apply any case-specific functions
        #self.InputInpFunction(i)
        # Loop through input sequence
        for j in self.opts.get_PhaseSequence():
            # Set the "restart_read" property appropriately
            # Get the reduced namelist for sequence *j*
            iopts = self.opts.select_InputInp(j)
            # Do simple blocks
            for block in ["CFD_SOLVER", "CFD_SOLVER_OPTS", "TAILOR"]:
                # Get options
                blockopts = iopts.get(block, {})
                # Check if it's a dict
                if not isinstance(blockopts, dict):
                    continue
                # Name of setter function
                settername = "set_%s_key" % block.replace("_", "")
                # Get setter
                fn = inp.__class__.__dict__.get(settername)
                # Make sure setter exists
                if not callable(fn):
                    raise KeyError(
                        "No recognized 'input.inp' block '%s'" % block)
                # Loop through settings
                for key, opt in blockopts.items():
                    # Apply the setting (may result in errors!
                    fn(inp, key, opt)
                # TODO: "MANAGE" block

            # Set number of iterations
            # ...
            # Write processed file in the case folder
            fout = os.path.join(frun, 'input.%02i.inp' % j)
            # Write the input file.
            inp.Write(fout)
        # Return to original path.
        os.chdir(fpwd)

    # Prepare freestream conditions
    def PrepareInputInpFlightConditions(self, i):
        r"""Set flight conditions in ``input.inp``

        :Call:
            >>> cntl.PrepareInputInpFlightConditions(i)
        :Inputs:
            *cntl*: :class:`cape.pyus.cntl.Cntl`
                Instance of US3D control class
            *i*: :class:`int`
                Run index
        :Versions:
            * 2018-04-19 ``@ddalle``: First version
            * 2020-04-13 ``@ddalle``: Forked from :mod:`cape.pyfun`
        """
        # Unpack handles
        x = self.x
        inp = self.InputInp
        # Get name(s) of BC for freestream
        compIDs = self.GetInflowBC()
        # Get flow angle properties
        a  = x.GetAlpha(i)
        b  = x.GetBeta(i)
        # Angle of attack
        if a is not None:
            for comp in compIDs:
                inp.SetAlpha(a, name=comp)
        # Sideslip angle
        if b is not None:
            for comp in compIDs:
                inp.SetBeta(b, name=comp)
        # Get state properties
        rho = x.GetDensity(i)
        T = x.GetTemperature(i)
        V = x.GetVelocity(i)
        # Check for vibrational temperature key
        ktv = x.GetKeysByType("Tv")
        # If any, use that
        if len(ktv) > 0:
            # Get vibrational temperature
            Tv = x[ktv[0]][i]
        else:
            # Use static temperature
            Tv = T
        # Set density
        if rho is not None:
            for comp in compIDs:
                inp.SetDensity(rho, name=comp)
        # Set temperature
        if T is not None:
            for comp in compIDs:
                inp.SetTemperature(T, name=comp)
        # Set vibrational temperature
        if Tv is not None:
            for comp in compIDs:
                inp.SetVibTemp(Tv, name=comp)
        # Set velocity
        if V is not None:
            for comp in compIDs:
                inp.SetVelocity(V, name=comp)

    # Get name of BC used for freestream
    def GetInflowBC(self):
        r"""Find name(s) of freestream BC faces

        :Call:
            >>> compIDs = cntl.GetInflowBC()
        :Inputs:
            *cntl*: :class:`cape.pyus.cntl.Cntl`
                Instance of US3D control class
        :Outputs:
            *compIDs*: :class:`list`\ [:class:`str`]
                List of 1 or more BC names
        :Versions:
            * 2020-04-28 ``@ddalle``: First version
        """
        # Trajectory handle
        x = self.x
        # Valid flight condition key *Type*\ s
        ktypes = {
            "mach", "alpha", "beta", "alpha_t", "phi",
            "T", "rho", "V", "p", "p0", "q", "Tv", "T0", "Re"
        }
        # Loop through actual run matrix keys
        for col in x.cols:
            # Get definition
            defn = x.defns.get(col)
            # Check for valid definition
            if defn is None:
                continue
            # Get type
            if defn.get("Type") not in ktypes:
                continue
            # Get CompID
            compID = defn.get("CompID")
            # Check for valid CompID
            if compID is not None:
                break
        else:
            # If None found, try finding from *InputInp* interface
            inp = self.InputInp
                # Process existing BCs
            bcs = inp.ReadBCs()
            # Get *bcn* for existing BCs
            if isinstance(bcs, dict):
                # Process the :class:`dict`
                bcns = {bc.get("bcn"): name for name, bc in bcs.items()}
            else:
                # Empty BCs
                bcns = {}
            # Get name of BC with *bcn* of ``10``
            if 10 in bcns:
                # Get it (this will be the last one if multiple
                compID = bcns[10]
            else:
                # Just use a standard name
                compID = "inflow"
        # Ensure list
        if isinstance(compID, list):
            # Already list
            compIDs = compID
        else:
            # Make list; "inflow" -> ["inflow"]
            compIDs = [compID]
        # Output
        return compIDs
   # ]
  # >

  # ======
  # Mesh
  # ======
  # <
    # Get list of raw file names
    def GetInputMeshFileNames(self):
        r"""Return the list of mesh files from file

        :Call:
            >>> fname = cntl.GetInputMeshFileNames()
        :Inputs:
            *cntl*: :class:`cape.pyus.cntl.Cntl`
                Instance of control class
        :Outputs:
            *fname*: :class:`list`\ [:class:`str`]
                List of file names read from root directory
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
            * 2020-04-13 ``@ddalle``: Forked from :mod:`cape.pyfun`
        """
        # Get the file names from *opts*
        fname = self.opts.get_MeshFile()
        # Ensure list
        if fname is None:
            # Remove ``None``
            return []
        elif isinstance(fname, (list, np.ndarray, tuple)):
            # Return list-like as list (copied)
            return list(fname)
        else:
            # Convert to list.
            return [fname]

    # Get list of mesh file names that should be in a case folder.
    def GetProcessedMeshFileNames(self):
        r"""Return the list of mesh files that are written

        :Call:
            >>> fname = cntl.GetProcessedMeshFileNames()
        :Inputs:
            *cntl*: :class:`cape.pyus.cntl.Cntl`
                Instance of control class
        :Outputs:
            *fname*: :class:`list`\ [:class:`str`]
                List of file names written to case folders
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
            * 2020-04-13 ``@ddalle``: Forked from :mod:`cape.pyfun`
        """
        # Initialize output
        fname = []
        # Loop through input files.
        for f in self.GetInputMeshFileNames():
            # Get processed name
            fname.append(self.ProcessMeshFileName(f))
        # Output
        return fname

    # Process a mesh file name to use the project root name
    def ProcessMeshFileName(self, fname):
        r"""Return a mesh file name using the project root name

        :Call:
            >>> fout = cntl.ProcessMeshFileName(fname)
        :Inputs:
            *cntl*: :class:`cape.pyus.cntl.Cntl`
                Instance of control class
            *fname*: :class:`str`
                Raw file name to be converted to case-folder file name
        :Outputs:
            *fout*: :class:`str`
                Name of file name using project name as prefix
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
            * 2020-04-13 ``@ddalle``: Forked from :mod:`cape.pyfun`
        """
        # Get project name (hard-coded for now)
        fproj = self.GetProjectRootName()
        # Special name extensions
        ffmt = ["b8", "lb8", "b4", "lb4", "r8", "lr8", "r4", "lr4"]
        # Get extension
        fsplt = fname.split('.')
        fext = fsplt[-1]
        # Use project name plus the same extension.
        if len(fsplt) > 1 and fsplt[-2] in ffmt:
            # Copy second-to-last extension
            return "%s.%s.%s" % (fproj, fsplt[-2], fext)
        else:
            # Just the extension
            return "%s.%s" % (fproj, fext)

    # Function to check if the mesh for case *i* is prepared
    def CheckMesh(self, i):
        r"""Check if the mesh for case *i* is prepared

        :Call:
            >>> q = cntl.CheckMesh(i)
        :Inputs:
            *fun3d*: :class:`cape.pyus.cntl.Cntl`
                Instance of control class
            *i*: :class:`int`
                Index of the case to check
        :Outputs:
            *q*: :class:`bool`
                Whether or not the mesh for case *i* is prepared
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
            * 2020-04-13 ``@ddalle``: Forked from :mod:`cape.pyfun`
        """
        # Check input
        if not isinstance(i, int):
            raise TypeError("Case index must be an integer")
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
            *fun3d*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class containing relevant parameters
            *v*: ``True`` | {``False``}
                Verbose flag
        :Outputs:
            *q*: ``True`` | ``False``
                Whether current folder has the required mesh files
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
            * 2017-02-22 ``@ddalle``: Added verbose option
            * 2020-04-13 ``@ddalle``: Forked from :mod:`cape.pyfun`
        """
        # Get list of mesh file names
        fmesh = self.GetProcessedMeshFileNames()
        # Check for presence
        for f in fmesh:
            # Check for the file
            q = os.path.isfile(f)
            # Exit otherwise
            if not q:
                # Verbose option
                if v:
                    print("    Missing mesh file '%s'" % fmesh)
                # Terminate
                return False
        # All files found
        return True
  # >

  # ================
  # File: input.inp
  # ================
  # <
    # Read the namelist
    def ReadInputInp(self, j=0, q=True):
        r"""Read the ``input.inp`` template file

        :Call:
            >>> cntl.InputInp(j=0, q=True)
        :Inputs:
            *cntl*: :class:`cape.pyus.us3d.US3D`
                US3D control interface
            *j*: :class:`int`
                Phase number
            *q*: {``True``} | ``False``
                Whether or not to read to *Namelist*, else *Namelist0*
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
            * 2015-12-31 ``@ddalle``: Added *Namelist0*
            * 2019-06-27 ``@ddalle``: Forked from :func:`ReadNamelist`
        """
        # template file name
        finp = self.opts.get_InputInp(j)
        # Check for empty value
        if finp is None:
            return
        # Check for absolute path
        if not os.path.isabs(finp):
            # Use path relative to JSON root
            finp = os.path.join(self.RootDir, finp)
        # Read the file
        inp = InputInp(finp)
        # Save it.
        if q:
            # Read to main slot for modification
            self.InputInp = inp
        else:
            # Template for reading original parameters
            self.NamelisInputInp0 = inp

    # Get value from ``input.inp`` key
    def GetInputInpVar(self, sec, key, j=0):
        r"""Get a ``input.inp`` variable's value

        The JSON file overrides the value from the template file

        :Call:
            >>> val = cntl.getInputVar(sec, key, j=0)
        :Inputs:
            *cntl*: :class:`cape.pyus.us3d.US3D`
                US3D control interface
            *sec*: :class:`str`
                Name of namelist section
            *key*: :class:`str`
                Variable to read
            *j*: :class:`int`
                Run sequence index
        :Outputs:
            *val*: :class:`any`
                Value
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
            * 2019-06-27 ``@ddalle``: From :func:`GetNamelistVar`
        """
        # Get the value from template
        tval = self.InputInp.GetVar(sec, key)
        # Check for options value.
        if tval is None:
            # No template file value
            return self.opts.get_InputInp_key(sec, key, j)
        elif 'US3D' not in self.opts:
            # No section in options
            return tval
        elif sec not in self.opts['US3D']:
            # No corresponding options section
            return tval
        elif key not in self.opts['US3D'][sec]:
            # Value not specified in the options namelist
            return tval
        else:
            # Default to the options
            return self.opts.get_InputInp_key(sec, key, j)
  # >

  # ============
  # Case Status
  # ============
  # <
    # Get total CPU hours (actually core hours)
    def GetCPUTime(self, i, running=False):
        r"""Read a CAPE-style core-hour file from a case

        :Call:
            >>> CPUt = cntl.GetCPUTime(i, running=False)
        :Inputs:
            *cntl*: :class:`cape.pyus.us3d.US3D`
                US3D control interface
            *i*: :class:`int`
                Case index
            *runing*: ``True`` | {``False``}
                Whether or not to check for time since last start
        :Outputs:
            *CPUt*: :class:`float` | ``None``
                Total core hours used in this job
        :Versions:
            * 2015-12-22 ``@ddalle``: First version
            * 2016-08-31 ``@ddalle``: Checking time since last start
        """
        # File names
        fname = 'pyus_time.dat'
        fstrt = 'pyus_start.dat'
        # Call the general function using hard-coded file name
        return self.GetCPUTimeBoth(i, fname, fstrt, running=running)
  # >

  # ==============
  # Case Interface
  # ==============
  # <
    # Get the project rootname
    def GetProjectRootName(self, j=0):
        r"""Get the project root name

        Right now this is just hard-coded

        :Call:
            >>> name = cntl.GetProjectName(j=0)
        :Inputs:
            *fun3d*: :class:`cape.pyfun.cntl.Cntl`
                Instance of global pyFun settings object
            *j*: :class:`int`
                Phase number
        :Outputs:
            *name*: :class:`str`
                Project root name
        :Versions:
            * 2020-04-13 ``@ddalle``: First version
        """
        # Use hard-coded
        return "pyus"

    # Read run control options from case JSON file
    def ReadCaseJSON(self, i):
        """Read ``case.json`` file from case *i* if possible

        :Call:
            >>> rc = cntl.ReadCaseJSON(i)
        :Inputs:
            *cntl*: :class:`cape.pyus.us3d.US3D`
                US3D control interface
            *i*: :class:`int`
                Run index
        :Outputs:
            *rc*: ``None`` | :class:`cape.pyus.options.runControl.RunControl`
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

    # Read ``input.inp`` file from a case folder
    def ReadCaseInputInp(self, i, rc=None, j=None):
        r"""Read ``input.inp`` from case *i*, phase *j* if possible

        :Call:
            >>> inp = cntl.ReadCaseInputInp(i, rc=None, j=None)
        :Inputs:
            *cntl*: :class:`cape.pyus.us3d.US3D`
                US3D control interface
            *i*: :class:`int`
                Case index
            *rc*: ``None`` | :class:`cape.pyus.options.runControl.RunControl`
                Run control interface read from ``case.json`` file
            *j*: {``None``} | nonnegative :class:`int`
                Phase number
        :Outputs:
            *inp*: ``None`` | :class:`cape.pyus.inputInp.InputInp`
                US3D input interface if possible
        :Versions:
            * 2016-12-12 ``@ddalle``: First version
            * 2019-06-27 ``@ddalle``: From cape.pyus
        """
        # Read the *rc* if necessary
        if rc is None:
            rc = self.ReadCaseJSON(i)
        # If still None, exit
        if rc is None:
            return
        # Get phase number
        if j is None:
            j = rc.get_PhaseSequence(-1)
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
        # Read the namelist
        nml = case.GetInputInp(rc, j)
        # Return to original location
        os.chdir(fpwd)
        # Output
        return nml

    # Write the PBS script.
    def WritePBS(self, i):
        r"""Write the PBS script(s) for a given case

        :Call:
            >>> cntl.WritePBS(i)
        :Inputs:
            *cntl*: :class:`cape.pyus.us3d.US3D`
                US3D control interface
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014-10-19 ``@ddalle``: First version
            * 2019-06-27 ``@ddalle``: US3D version
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
                fpbs = 'run_us3d.%02i.pbs' % j
            else:
                # Use single PBS script with plain name.
                fpbs = 'run_us3d.pbs'
            # Initialize the PBS script.
            f = open(fpbs, 'w')
            # Write the header.
            self.WritePBSHeader(f, i, j)

            # Initialize options to `run_FUN3D.py`
            flgs = ''

            # Simply call the advanced interface.
            f.write('\n# Call the US3D interface.\n')
            f.write('run_us3d.py' + flgs + '\n')

            # Close the file.
            f.close()
        # Return.
        os.chdir(fpwd)

    # Call the correct :mod:`case` module to start a case
    def CaseStartCase(self):
        r"""Start a case by either submitting it or running it

        This function relies on :mod:`cape.pyus.case`, and so it is
        customized for the US3D solver only in that it calls the correct
        :mod:`case` module.

        :Call:
            >>> pbs = cntl.CaseStartCase()
        :Inputs:
            *cntl*: :class:`cape.pyus.us3d.US3D`
                US3D control interface
        :Outputs:
            *pbs*: :class:`int` or ``None``
                PBS job ID if submitted successfully
        :Versions:
            * 2015-10-14 ``@ddalle``: First version
            * 2019-06-27 ``@ddalle``: US3D version
        """
        return case.StartCase()
  # >
