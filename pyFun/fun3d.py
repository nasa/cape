"""
FUN3D control module: :mod:`pyFun.fun3d`
========================================

This module provides tools to interface with most FUN3D settings from Python.
"""

# Basic numerics
import numpy as np
# Configuration file processor
import json
# Date processing
from datetime import datetime
# File system and operating system management
import os, shutil
import subprocess as sp

# Import template class
from cape.cntl import Cntl

# Local classes
from namelist   import Namelist
from rubberData import RubberData

# Other pyFun modules
from . import options
from . import case
from . import mapbc
# Unmodified CAPE modules
from cape import convert

# Functions and classes from other modules
from trajectory import Trajectory

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyFunFolder = os.path.split(_fname)[0]
    
# Class to read input files
class Fun3d(Cntl):
    """
    Class for handling global options and setup for FUN3D.
    
    This class is intended to handle all settings used to describe a group
    of Cart3D cases.  For situations where it is not sufficiently
    customized, it can be used partially, e.g., to set up a Mach/alpha sweep
    for each single control variable setting.
    
    The settings are read from a JSON file, which is robust and simple to
    read, but has the disadvantage that there is no support for comments.
    Hopefully the various names are descriptive enough not to require
    explanation.
    
    Defaults are read from the file ``$CAPE/settings/pyFun.default.json``.
    
    :Call:
        >>> fun3d = pyFun.Fun3d(fname="pyFun.json")
    :Inputs:
        *fname*: :class:`str`
            Name of pyFun input file
    :Outputs:
        *fun3d*: :class:`pyFun.fun3d.Fun3d`
            Instance of the pyFun control class
    :Data members:
        *fun3d.opts*: :class:`dict`
            Dictionary of options for this case (directly from *fname*)
        *fun3d.x*: :class:`pyFun.trajectory.Trajectory`
            Values and definitions for variables in the run matrix
        *fun3d.RootDir*: :class:`str`
            Absolute path to the root directory
    :Versions:
        * 2015-10-16 ``@ddalle``: Started
    """ 
    # Initialization method
    def __init__(self, fname="pyFun.json"):
        """Initialization method for :mod:`cape.cntl.Cntl`"""
        
        # Read settings
        self.opts = options.Options(fname=fname)
        
        #Save the current directory as the root
        self.RootDir = os.getcwd()
        
        # Import modules
        self.ImportModules()
        
        # Process the trajectory.
        self.x = Trajectory(**self.opts['Trajectory'])

        # Job list
        self.jobs = {}
        
        # Read the namelist.
        self.ReadNamelist()
        
        # Check for dual
        if self.opts.get_Dual():
            self.ReadRubberData()
        
        # Read the boundary conditions
        self.ReadMapBC()
        
        # Set umask
        os.umask(self.opts.get_umask())
        
    # Output representation
    def __repr__(self):
        """Output representation for the class."""
        # Display basic information from all three areas.
        return "<pyFun.Fun3d(nCase=%i)>" % (
            self.x.nCase)
        
    # Read the boundary condition map
    def ReadMapBC(self, j=0, q=True):
        """Read the FUN3D boundary condition map
        
        :Call:
            >>> fun3d.ReadMapBC(q=True)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of the pyFun control class
            *q*: {``True``} | ``False``
                Whether or not to read to *MapBC*, else *MapBC0*
        :Versions:
            * 2016-03-30 ``@ddalle``: First version
        """
        # Change to root safely
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Read the file
        BC = mapbc.MapBC(self.opts.get_MapBCFile(j))
        # Save it.
        if q:
            # Read to main slot.
            self.MapBC = BC
        else:
            # Template
            self.MapBC0 = BC
        # Go back to original location
        os.chdir(fpwd)
        
    # Read the namelist
    def ReadNamelist(self, j=0, q=True):
        """Read the :file:`fun3d.nml` file
        
        :Call:
            >>> fun3d.ReadInputCntl(j=0, q=True)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of the pyFun control class
            *j*: :class:`int`
                Phase number
            *q*: :class:`bool`
                Whether or not to read to *Namelist*, else *Namelist0*
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
            * 2015-12-31 ``@ddalle``: Added *Namelist0*
        """
        # Change to root safely.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Read the file.
        nml = Namelist(self.opts.get_Namelist(j))
        # Save it.
        if q:
            # Read to main slot for modification
            self.Namelist = nml
        else:
            # Template for reading original parameters
            self.Namelist0 = nml
        # Go back to original location
        os.chdir(fpwd)
        
    # Read the ``rubber.data`` file
    def ReadRubberData(self, j=0, q=True):
        """Read the :file:`rubber.data` file
        
        :Call:
            >>> fun3d.ReadRubberData(j=0, q=True)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of the pyFun control class
            *j*: :class:`int`
                Phase number
            *q*: :class:`bool`
                Whether or not read *RubberData*, else *RubberData0*
        :Versions:
            * 2016-04-27 ``@ddalle``: First version
        """
        # Change to root safely
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
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
        # Go back to original location
        os.chdir(fpwd)
        
    # Get namelist var
    def GetNamelistVar(self, sec, key, j=0):
        """Get a namelist variable's value
        
        The JSON file overrides the value from the namelist file
        
        :Call:
            >>> val = fun3d.GetNamelistVar(sec, key, j=0)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of global pyFun settings object
            *sec*: :class:`str`
                Name of namelist section
            *key*: :class:`str`
                Variable to read
            *j*: :class:`int`
                Run sequence index
        :Outputs:
            *val*: :class:`int` | :class:`float` | :class:`str` | :class:`list`
                Value
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
        """
        # Get the namelist value.
        nval = self.Namelist.GetVar(sec, key)
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
            return self.opts_get_namelist_var(sec, key, i)
        
    # Get the project rootname
    def GetProjectRootName(self, j=0):
        """Get the project root name
        
        The JSON file overrides the value from the namelist file if appropriate
        
        :Call:
            >>> name = fun3d.GetProjectName(j=0)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of global pyFun settings object
            *j*: :class:`int`
                Phase number
        :Outputs:
            *name*: :class:`str`
                Project root name
        :Versions:
            * 2015-10-18 ``@ddalle``: First version
        """
        # Read the namelist.
        self.ReadNamelist(j, False)
        # Get the namelist value.
        nname = self.Namelist0.GetVar('project', 'project_rootname')
        # Get the options value.
        oname = self.opts.get_project_rootname(j)
        # Check for options value
        if nname is None:
            # Use the options value.
            name = oname
        elif 'Fun3D' not in self.opts:
            # No namelist options
            name = nname
        elif 'project' not in self.opts['Fun3D']:
            # No project options
            name = nname
        elif 'project_rootname' not in self.opts['Fun3D']['project']:
            # No rootname
            name = nname
        else:
            # Use the options value.
            name = oname
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
        """Get the grid format
        
        The JSON file overrides the value from the namelist file
        
        :Call:
            >>> fmt = fun3d.GetGridFormat(j=0)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of global pyFun settings object
            *j*: :class:`int`
                Run sequence index
        :Outputs:
            *fmt*: :class:`str`
                Project root name
        :Versions:
            * 2015-10-18 ``@ddalle``: First version
        """
        return self.GetNamelistVar('raw_grid', 'grid_format', j)
            

    # Get the current iteration number from :mod:`case`
    def CaseGetCurrentIter(self):
        """Get the current iteration number from the appropriate module
        
        This function utilizes the :mod:`cape.case` module, and so it must be
        copied to the definition for each solver's control class
        
        :Call:
            >>> n = fun3d.CaseGetCurrentIter()
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *n*: :class:`int` or ``None``
                Number of completed iterations or ``None`` if not set up
        :Versions:
            * 2015-10-14 ``@ddalle``: First version
        """
        # Read value
        n = case.GetCurrentIter()
        # Default to zero.
        if n is None:
            return 0
        else:
            return n
        
    # Get last iter
    def GetLastIter(self, i):
        """Get minimum required iteration for a given run to be completed
        
        :Call:
            >>> nIter = fun3d.GetLastIter(i)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Outputs:
            *nIter*: :class:`int`
                Number of iterations required for case *i*
        :Versions:
            * 2014-10-03 ``@ddalle``: First version
        """
        # Check the case
        if self.CheckCase(i) is None:
            return None
        # Safely go to root directory.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Go there.
        os.chdir(frun)
        # Read the local case.json file.
        rc = case.ReadCaseJSON()
        # Return to original location.
        os.chdir(fpwd)
        # Output
        return rc.get_LastIter()
        
        
    # Get list of raw file names
    def GetInputMeshFileNames(self):
        """Return the list of mesh files from file
        
        :Call:
            >>> fname = fun3d.GetInputMeshFileNames()
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
        :Outputs:
            *fname*: :class:`list` (:class:`str`)
                List of file names read from root directory
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
        """
        # Get the file names from *opts*
        fname = self.opts.get_MeshFile()
        # Ensure list.
        if type(fname).__name__ not in ['list', 'ndarray']:
            # Convert to list.
            return [fname]
        else:
            # Return output
            return fname
        
    # Get list of mesh file names that should be in a case folder.
    def GetProcessedMeshFileNames(self):
        """Return the list of mesh files that are written
        
        :Call:
            >>> fname = fun3d.GetProcessedMeshFileNames()
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
        :Outputs:
            *fname*: :class:`list` (:class:`str`)
                List of file names written to case folders
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
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
        """Return a mesh file name using the project root name
        
        :Call:
            >>> fout = fun3d.ProcessMeshFileName(fname)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
            *fname*: :class:`str`
                Raw file name to be converted to case-folder file name
        :Outputs:
            *fout*: :class:`str`
                Name of file name using project name as prefix
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        # Get project name
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
        """Check if the mesh for case *i* is prepared
        
        :Call:
            >>> q = fun3d.CheckMesh(i)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check
        :Outputs:
            *q*: :class:`bool`
                Whether or not the mesh for case *i* is prepared
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
        """
        # Check input
        if not type(i).__name__.startswith("int"):
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
    def CheckMeshFiles(self):
        """Check for the mesh files in the present folder
        
        :Call:
            >>> q = fun3d.CheckMeshFiles()
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
        :Outputs:
            *q*: :class:`bool`
                Whether or not the present folder has the required mesh files
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Initialize status
        q = True
        # Get list of mesh file names
        fmesh = self.GetProcessedMeshFileNames()
        # Check for presence
        for f in fmesh:
            # Check for the file
            q = q and os.path.isfile(f)
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
            else:
                # No surface or mesh files
                q = False
        # Output
        return q
        
    # Prepare the mesh for case *i* (if necessary)
    def PrepareMesh(self, i):
        """Prepare the mesh for case *i* if necessary
        
        :Call:
            >>> fun3d.PrepareMesh(i)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
        """
        # ---------
        # Case info
        # ---------
        # Check the mesh.
        if self.CheckMesh(i):
            return None
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Get the name of the group.
        fgrp = self.x.GetGroupFolderNames(i)
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
        # Check for groups with common meshes.
        if self.opts.get_GroupMesh():
            # Get the group index.
            j = self.x.GetGroupIndex(i)
            # Status update
            print("  Group name: '%s' (index %i)" % (fgrp,j))
            # Enter the group folder.
            os.chdir(fgrp)
        else:
            # Check if the fun folder exists.
            if not os.path.isdir(frun):
                self.mkdir(frun)
            # Status update
            print("  Case name: '%s' (index %i)" % (frun,i))
            # Enter the case folder.
            os.chdir(frun)
            # Do we need "Adapt" and "Flow" folders?
            if self.opts.get_Dual():
                # Create folder for the primal solution
                if not os.path.isdir('Flow'):    self.mkdir('Flow')
                if not os.path.isdir('Adjoint'): self.mkdir('Adjoint')
                # Enter
                os.chdir('Flow')
        # ----------
        # Copy files
        # ----------
        # Get the names of the raw input files and target files
        finp = self.GetInputMeshFileNames()
        fmsh = self.GetProcessedMeshFileNames()
        # Project name
        fproj = self.GetProjectRootName(0)
        # Loop through those files
        for j in range(len(finp)):
            # Original and final file names
            f0 = os.path.join(self.RootDir, finp[j])
            f1 = fmsh[j]
            # Copy fhe file.
            if os.path.isfile(f0):
                shutil.copyfile(f0, f1)
        # ------------------
        # Triangulation prep
        # ------------------
        # Check for triangulation
        if self.opts.get_aflr3():
            # Status update
            print("  Preparing surface triangulation...")
            # Read the mesh.
            self.ReadTri()
            # Revert to initial surface.
            self.tri = self.tri0.Copy()
            # Apply rotations, translations, etc.
            self.PrepareTri(i)
            # AFLR3 boundary conditions file
            fbc = self.opts.get_aflr3_BCFile()
            # Check for those AFLR3 boundary conditions
            if fbc:
                # Absolute file name
                fbc = os.path.join(self.RootDir, fbc)
                # Copy the file
                shutil.copyfile(fbc, '%s.aflr3bc' % fproj)
            # Check intersection status.
            if self.opts.get_intersect():
                # Write tri file as non-intersected; each volume is one CompID
                self.tri.WriteVolTri('%s.tri' % fproj)
                # Write the existing triangulation with existing CompIDs.
                self.tri.Write('%s.c.tri' % fproj)
            elif self.opts.get_verify():
                # Write the tri file
                self.tri.Write('%s.i.tri' % fproj)
                # Write the AFLR3 surface file
                self.tri.WriteSurf('%s.surf' % fproj)
            else:
                # Write the AFLR3 surface file only
                self.tri.WriteSurf('%s.surf' % fproj)
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
            exec("%s(self.%s,i=%i)" % (func, getattr(self.x,key)[i], i))
        # Check for jumpstart.
        if self.opts.get_PreMesh(0) and self.opts.get_aflr3():
            # Run ``intersect`` if appropriate
            case.CaseIntersect(rc, fproj, 0)
            # Run ``verify`` if appropriate
            case.CaseVerify(rc, fproj, 0)
            # Create the mesh if appropriate
            case.CaseAFLR3(rc, proj=fproj, fmt=self.nml.GetGridFormat(), n=0)
        # -------
        # Cleanup
        # -------
        # Return to original folder
        os.chdir(fpwd)
        
        
    # Check if cases with zero iterations are not yet setup to run
    def CheckNone(self):
        """Check if the current folder has the necessary files to run
        
        :Call:
            >>> q = fun3d.CheckNone()
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
            * 2016-04-11 ``@ddalle``: Checking for AFLR3 input files, too
            * 2016-04-29 ``@ddalle``: Simpler version that handles ``Flow/``
        """
        # Settings file.
        if not os.path.isfile('case.json'): return True
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
                if qdual: os.chdir('..')
                return False
        # Namelist file
        if not os.path.isfile('fun3d.00.nml'):
            if qdual: os.chdir('..')
            return True
        # Check mesh files
        q = self.CheckMeshFiles()
        # Go back if appropriate
        if qdual: os.chdir('..')
        # Output
        return not q
            
    # Get total CPU hours (actually core hours)
    def GetCPUTime(self, i):
        """Read a CAPE-style core-hour file from a case
        
        :Call:
            >>> CPUt = fun3d.GetCPUTime(i)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                FUN3D control interface
            *i*: :class:`int`
                Case index
        :Outputs:
            *CPUt*: :class:`float` | ``None``
                Total core hours used in this job
        :Versions:
            * 2015-12-22 ``@ddalle``: First version
        """
        # Call the general function using hard-coded file name
        return self.GetCPUTimeFromFile(i, fname='pyfun_time.dat')

    
    # Prepare a case.
    def PrepareCase(self, i):
        """Prepare a case for running if it is not already prepared
        
        :Call:
            >>> fun3d.PrepareCase(i)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of case to prepare/analyze
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
        """
        # Get the existing status.
        n = self.CheckCase(i)
        # Quit if already prepared.
        if n is not None: return
        # Prepare the mesh (and create folders if necessary).
        self.PrepareMesh(i)
        # Check for dual
        qdual = self.opts.get_Dual()
        # Get the run name.
        frun = self.x.GetFullFolderNames(i)
        # Go to root folder safely.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Enter the run directory.
        if not os.path.isdir(frun): self.mkdir(frun)
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
                if os.path.isfile(fto): os.remove(fto)
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
        if qdual: self.ReadRubberData()
        # Loop through the functions.
        for (key, func) in zip(keys, funcs):
            # Apply it.
            exec("%s(self,%s,i=%i)" % (func, getattr(self.x,key)[i], i))
        # Prepare the rubber.data file
        self.PrepareRubberData(i)
        # Write the fun3d.nml file(s).
        self.PrepareNamelist(i)
        # Write a JSON file with
        self.WriteCaseJSON(i)
        # Write the PBS script.
        self.WritePBS(i)
        # Return to original location
        os.chdir(fpwd)
        
        
    # Function to prepare "input.cntl" files
    def PrepareNamelist(self, i):
        """
        Write :file:`fun3d.nml` for run case *i* in the appropriate folder
        and with the appropriate settings.
        
        :Call:
            >>> fun3d.PrepareNamelist(i)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of FUN3D control class
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014-06-04 ``@ddalle``: First version
            * 2014-06-06 ``@ddalle``: Low-level functionality for grid folders
            * 2014-09-30 ``@ddalle``: Changed to write only a single case
        """
        # Read namelist file
        self.ReadNamelist()
        # Extract trajectory.
        x = self.x
        # Go safely to root folder.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Set the flight conditions.
        # Mach number
        M = x.GetMach(i)
        if M  is not None: self.Namelist.SetMach(M)
        # Angle of attack
        a = x.GetAlpha(i)
        if a  is not None: self.Namelist.SetAlpha(a)
        # Sideslip angle
        b = x.GetBeta(i)
        if b  is not None: self.Namelist.SetBeta(b)
        # Reynolds number
        Re = x.GetReynoldsNumber(i)
        if Re is not None: self.Namelist.SetReynoldsNumber(Re)
        # Temperature
        T = x.GetTemperature(i)
        if T  is not None: self.Namelist.SetTemperature(T)
        
        # Get the case.
        frun = self.x.GetFullFolderNames(i)
        # Set up the component force & moment tracking
        self.PrepareNamelistConfig()
        
        # Set the surface BCs
        for k in self.x.GetKeysByType('SurfBC'):
            # Ensure the presence of the triangulation
            self.ReadTri()
            # Apply the appropriate methods
            self.SetSurfBC(k, i)
        # Set the surface BCs that use thrust as input
        for k in self.x.GetKeysByType('SurfCT'):
            # Ensure the presence of the triangulation
            self.ReadTri()
            # Apply the appropriate methods
            self.SetSurfBC(k, i, CT=True)
        # File name
        if self.opts.get_Dual():
            # Write in the 'Flow/' folder
            fout = os.path.join(frun, 'Flow', 
                '%s.mapbc' % self.GetProjectRootName(0))
        else:
            # Main folder
            fout = os.path.join(frun, '%s.mapbc'%self.GetProjectRootName(0))
        # Write the BC file
        self.MapBC.Write(fout)
        
        # Make folder if necessary.
        if not os.path.isdir(frun): self.mkdir(frun)
        # Loop through input sequence
        for j in range(self.opts.get_nSeq()):
            # Set the "restart_read" property appropriately
            # This setting is overridden by *nopts* if appropriate
            if j == 0:
                # First run sequence; not restart
                self.Namelist.SetVar('code_run_control', 'restart_read', 'off')
            else:
                # Later sequence; restart
                self.Namelist.SetVar('code_run_control', 'restart_read', 'on')
            # Set number of iterations
            self.Namelist.SetnIter(self.opts.get_nIter(j))
            # Get the reduced namelist for sequence *j*
            nopts = self.opts.select_namelist(j)
            dopts = self.opts.select_dual_namelist(j)
            # Apply them to this namelist
            self.Namelist.ApplyDict(nopts)
            # Ensure correct *project_rootname*
            self.Namelist.SetRootname(self.GetProjectRootName(j))
            # Check for adaptive phase
            if self.opts.get_Adaptive() and self.opts.get_AdaptPhase(j):
                # Set the project rootname of the next phase
                self.Namelist.SetAdaptRootname(self.GetProjectRootName(j+1))
                # Check for adaptive grid
                if self.opts.get_AdaptationNumber(j) > 0:
                    # Always AFLR3/stream
                    self.Namelist.SetVar('raw_grid', 'grid_format', 'aflr3')
                    self.Namelist.SetVar('raw_grid', 'data_format', 'stream')
            # Name of output file.
            if self.opts.get_Dual():
                # Write in the "Flow/" folder
                fout = os.path.join(frun, 'Flow', 'fun3d.%02i.nml' % j)
            else:
                # Write in the case folder
                fout = os.path.join(frun, 'fun3d.%02i.nml' % j)
            # Write the input file.
            self.Namelist.Write(fout)
            # Check for dual phase
            if self.opts.get_Dual() and self.opts.get_DualPhase(j):
                # Apply dual options
                self.Namelist.ApplyDict(dopts)
                # Write in the "Adjoint/" folder as well
                fout = os.path.join(frun, 'Flow', 'fun3d.dual.%02i.nml' % j)
                # Set restart flag appropriately
                if self.opts.get_AdaptationNumber(j) == 0:
                    # No restart read (of adjoint file)
                    self.Namelist.SetVar(
                        'code_run_control', 'restart_read', 'off')
                else:
                    # Restart read of adjoint
                    self.Namelist.SetVar(
                        'code_run_control', 'restart_read', 'on')
                    # Always AFLR3/stream
                    self.Namelist.SetVar('raw_grid', 'grid_format', 'aflr3')
                    self.Namelist.SetVar('raw_grid', 'data_format', 'stream')
                # Set the iteration count
                self.Namelist.SetnIter(self.opts.get_nIterAdjoint(j))
                # Set the adapt phase
                self.Namelist.SetVar('adapt_mechanics', 'adapt_project',
                    self.GetProjectRootName(j+1))
                # Write the adjoint namelist
                self.Namelist.Write(fout)
        # Return to original path.
        os.chdir(fpwd)
    
    # Prepare ``rubber.data`` file
    def PrepareRubberData(self, i):
        """Prepare ``rubber.data`` file if appropriate
        
        :Call:
            >>> fun3d.PrepareRubberData(i)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of FUN3D control class
            *i*: :class:`int`
                Run index
        :Versions:
            * 2016-04-27 ``@ddalle``: First version
        """
        # Check options
        if not self.opts.get_Dual(): return
        # Get list of adaptive coefficients.
        coeffs = self.opts.get_AdaptCoeffs()
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
            comp = self.opts.get_FuncCoeffCompID(coeff)
            # Check if already in the list
            if comp not in comps: 
                # Get the surface IDs
                comps[comp] = self.CompID2SurfID(comp)
            # Get component ID list
            surfs = comps[comp]
            # Get the option values for this coefficient
            typ = self.opts.get_FuncCoeffType(coeff)
            w   = self.opts.get_FuncCoeffWeight(coeff)
            t   = self.opts.get_FuncCoeffTarget(coeff)
            p   = self.opts.get_FuncCoeffPower(coeff)
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
        
        
    # Get surface ID numbers
    def CompID2SurfID(self, compID):
        """Convert triangulation component ID to surface index
        
        This relies on an XML configuration file and a FUN3D ``mapbc`` file
        
        :Call:
            >>> surfID = fun3d.CompID2SurfID(compID)
            >>> surfID = fun3d.CompID2SurfID(face)
            >>> surfID = fun3d.CompID2SurfID(comps)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of FUN3D control class
            *compID*: :class:`int`
                Surface boundary ID as used in surface mesh
            *face*: :class:`str`
                Name of face
            *comps*: :class:`list` (:class:`int` | :class:`str`)
                List of component IDs or face names
        :Outputs:
            *surfID*: :class:`list` (:class:`int`)
                List of corresponding indices of surface in MapBC
        :Versions:
            * 2016-04-27 ``@ddalle``: First version
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
        
    
    # Prepare surface BC
    def SetSurfBC(self, key, i, CT=False):
        """Set all surface BCs and flow initialization volumes for one key
        
        This uses the 7011 boundary condition and sets the values of BC
        stagnation pressure to freestream pressure and stagnation temperature to
        freestream temperature. Further, it creates a flow initialization volume
        to help with solution startup
        
        :Call:
            >>> fun3d.SetSurfBC(key, i, CT=False)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of global pyFun settings object
            *key*: :class:`str`
                Name of SurfBC key to process
            *i*: :class:`int`
                Case index
            *CT*: ``True`` | {``False``}
                Whether this key has thrust as input (else *p0*, *T0* directly)
        :Versions:
            * 2016-03-29 ``@ddalle``: First version
            * 2016-04-13 ``@ddalle``: Added SurfCT compatibility
        """
        # Get the BC inputs
        if CT:
            # Use thrust as input variable
            p0, T0 = self.GetSurfCTState(key, i)
        else:
            # Use *p0* and *T0* directly as inputs
            p0, T0 = self.GetSurfBCState(key, i)
        # Get the flow initialization volume state
        rho, U, a = self.GetSurfBCFlowInitState(key, i, CT=CT)
        # Get the namelist
        nml = self.Namelist
        # Get the components
        compIDs = self.x.GetSurfBC_CompID(i, key)
        # Current number of flow initialization volumes
        n = nml.GetNFlowInitVolumes()
        # Ensure list
        if type(compIDs).__name__ not in ['list', 'ndarray']:
            compIDs = [compIDs]
        # Boundary condition section
        sec = 'boundary_conditions'
        # Loop through the components
        for compID in compIDs:
            # Increase volume number
            n += 1
            # Convert to ID (if needed) and get the BC number to set
            compID = self.MapBC.GetCompID(compID)
            surfID = self.MapBC.GetSurfID(compID)
            # Set the BC to the correct value
            self.MapBC.SetBC(compID, 7011)
            # Set the BC
            nml.SetVar(sec, 'total_pressure_ratio',    p0, surfID)
            nml.SetVar(sec, 'total_temperature_ratio', T0, surfID)
            # Get the flow initialization volume
            x1, x2, r = self.GetSurfBCVolume(key, compID)
            # Get the surface normal
            N = self.tri.GetCompNormal(compID)
            # Velocity
            u = U * N[0]
            v = U * N[1]
            w = U * N[2]
            # Set the flow initialization state.
            nml.SetVar('flow_initialization', 'rho', rho,    n)
            nml.SetVar('flow_initialization', 'u',   U*N[0], n)
            nml.SetVar('flow_initialization', 'v',   U*N[1], n)
            nml.SetVar('flow_initialization', 'w',   U*N[2], n)
            nml.SetVar('flow_initialization', 'c',   a,      n)
            # Initialize the flow init vol
            nml.SetVar('flow_initialization', 'type_of_volume', 'cylinder', n)
            # Set the dimensions of the volume
            nml.SetVar('flow_initialization', 'radius', r, n)
            nml.SetVar('flow_initialization', 'point1', x1, (None,n))
            nml.SetVar('flow_initialization', 'point2', x2, (None,n))
        # Update number of volumes
        nml.SetNFlowInitVolumes(n)
    
    # Get surface BC inputs
    def GetSurfBCState(self, key, i):
        """Get stagnation pressure and temperature ratios
        
        :Call:
            >>> p0, T0 = fun3d.GetSurfBCState(key, i)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of global pyFun settings object
            *key*: :class:`str`
                Name of key to process
            *i*: :class:`int`
                Case index
        :Outputs:
            *p0*: :class:`float`
                Ratio of BC stagnation pressure to freestream static pressure
            *T0*: :class:`float`
                Ratio of BC stagnation temperature to freestream static temp
        :Versions:
            * 2016-03-29 ``@ddalle``: First version
        """
        # Get the inputs
        p0 = self.x.GetSurfBC_TotalPressure(i, key)
        T0 = self.x.GetSurfBC_TotalTemperature(i, key)
        # Calibration
        fp = self.x.GetSurfBC_PressureCalibration(i, key)
        # Reference pressure/temp
        pinf = self.x.GetSurfBC_RefPressure(i, key)
        Tinf = self.x.GetSurfBC_RefTemperature(i, key)
        # Output
        return fp*p0/pinf, T0/Tinf
        
    # Get surface CT state inputs
    def GetSurfCTState(self, key, i):
        """Get stagnation pressure and temperature ratios for *SurfCT* key
        
        :Call:
            >>> p0, T0 = fun3d.GetSurfCTState(key, i)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of global pyFun settings object
            *key*: :class:`str`
                Name of key to process
            *i*: :class:`int`
                Case index
        :Outputs:
            *p0*: :class:`float`
                Ratio of BC stagnation pressure to freestream static pressure
            *T0*: :class:`float`
                Ratio of BC stagnation temperature to freestream static temp
        :Versions:
            * 2016-04-13 ``@ddalle``: First version
        """
        # Get the thrust value
        CT = self.x.GetSurfCT_Thrust(i, key)
        # Get the exit parameters
        M2 = self.GetSurfCT_ExitMach(key, i)
        A2 = self.GetSurfCT_ExitArea(key, i)
        # Ratio of specific heats
        gam = self.x.GetSurfCT_Gamma(i, key)
        # Derivative gas constants
        g2 = 0.5 * (gam-1)
        g3 = gam / (gam-1)
        # Get reference dynamic pressure
        qref = self.x.GetSurfCT_RefDynamicPressure(i, key)
        # Get reference area
        Aref = self.GetSurfCT_RefArea(key, i)
        # Calculate total pressure
        p0 = CT*qref*Aref/A2 * (1+g2*M2*M2)**g3 / (1+gam*M2*M2)
        # Temperature inputs
        T0 = self.x.GetSurfCT_TotalTemperature(i, key)
        # Calibration
        fp = self.x.GetSurfCT_PressureCalibration(i, key)
        # Reference values
        pinf = self.x.GetSurfCT_RefPressure(i, key)
        Tinf = self.x.GetSurfCT_RefTemperature(i, key)
        qinf = self.x.GetSurfCT_RefDynamicPressure(i, key)
        # Output
        return fp*p0/pinf, T0/Tinf
        
    # Get startup volume for a surface BC input
    def GetSurfBCVolume(self, key, compID):
        """Get coordinates for flow initialization box
        
        :Call:
            >>> x1, x2, r = fun3d.GetSurfBCVolume(key, compID)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of global pyFun settings object
            *key*: :class:`str`
                Name of SurfBC key to process
            *compID*: :class:`int`
                Component ID for which to calculate flow volume
        :Outputs:
            *x1*: :class:`np.ndarray` (:class:`float`)
                First point of cylinder center line
            *x2*: :class:`np.ndarray` (:class:`float`)
                End point of cylinder center line
            *r*: :class:`float`
                Radius of cylinder
        :Versions:
            * 2016-03-29 ``@ddalle``: First version
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
    def GetSurfBCFlowInitState(self, key, i, CT=False):
        """Get nondimensional state for flow initialization volumes
        
        :Call:
            >>> rho, U, c = fun3d.GetSurfBCFlowInitState(key, i, CT=False)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of global pyFun settings object
            *key*: :class:`str`
                Name of SurfBC key to process
            *i*: :class:`int`
                Case index
            *CT*: ``True`` | {``False``}
                Whether this key has thrust as input (else *p0*, *T0* directly)
        :Outputs:
            *rho*: :class:`float`
                Normalized static density, *rho/rhoinf*
            *U*: :class:`float`
                Normalized velocity, *U/ainf*
            *c*: :class:`float`
                Normalized sound speed, *a/ainf*
        :Versions:
            * 2016-03-29 ``@ddalle``: First version
            * 2016-04-13 ``@ddalle``: Added *CT*/BC capability
        """
        # Get the boundary condition states
        if CT == True:
            # Use *SurfCT* thrust definition
            p0, T0 = self.GetSurfCTState(key, i)
        else:
            # Use *SurfBC* direct definitions of *p0*, *T0*
            p0, T0 = self.GetSurfBCState(key, i)
        # Get the Mach number and blending fraction
        M = self.x.defns[key].get('Mach', 0.2)
        f = self.x.defns[key].get('Blend', 0.9)
        # Ratio of specific heats
        gam = self.x.GetSurfBC_Gamma(i, key)
        # Calculate stagnation temperature ratio
        rT = 1 + (gam-1)/2*M*M
        # Stagnation-to-static ratios
        rr = rT ** (1/(gam-1))
        rp = rT ** (gam/(gam-1))
        # Reference values
        rho = f*(p0)/(T0) / rr
        c   = f*np.sqrt((T0) / rT)
        U   = M * c
        # Output
        return rho, U, c
        
    # Set up a namelist config
    def PrepareNamelistConfig(self):
        """Write the lines for the force/moment output in a namelist file
        
        :Call:
            >>> fun3d.PrepareNamelistConfig()
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
        :Versions:
            * 2015-10-20 ``@ddalle``: First version
        """
        # Get the components
        comps = self.opts.get_ConfigComponents()
        # Exit if no components
        if comps is None: return
        # Number
        n = len(comps)
        # Quit if nothing to do
        if n == 0: return
        # Extract namelist
        nml = self.Namelist
        # Loop through specified components.
        for k in range(1,n+1):
            # Get component.
            comp = comps[k-1]
            # Get input definitions.
            inp = self.opts.get_ConfigInput(comp)
            # Set input definitions.
            if inp is not None:
                nml.SetVar('component_parameters', 'component_input', inp, k)
            # Reference area
            if 'RefArea' in self.opts['Config']:
                # Get reference area.
                RefA = self.opts.get_RefArea(comp)
                # Set it
                nml.SetVar('component_parameters', 'component_sref', RefA, k)
            # Moment reference center
            if 'RefPoint' in self.opts['Config']:
                # Get MRP
                RefP = self.opts.get_RefPoint(comp)
                # Set the x- and y-coordinates
                nml.SetVar('component_parameters', 'component_xmc', RefP[0], k)
                nml.SetVar('component_parameters', 'component_ymc', RefP[1], k)
                # Check for z-coordinate
                if len(RefP) > 2:
                    nml.SetVar(
                        'component_parameters', 'component_zmc', RefP[2], k)
            # Reference length
            if 'RefLength' in self.opts['Config']:
                # Get reference length
                RefL = self.opts.get_RefLength(comp)
                # Set both reference lengths
                nml.SetVar('component_parameters', 'component_cref', RefL, k)
                nml.SetVar('component_parameters', 'component_bref', RefL, k)
            # Set the component name
            nml.SetVar('component_parameters', 'component_name', comp, k)
            # Tell FUN3D to determine the number of components on its own.
            nml.SetVar('component_parameters', 'component_count', -1, k)
        # Set the number of components
        nml.SetVar('component_parameters', 'number_of_components', n)
        
        
    # Write run control options to JSON file
    def WriteCaseJSON(self, i):
        """Write JSON file with run control and related settings for case *i*
        
        :Call:
            >>> fun3d.WriteCaseJSON(i)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
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
        # Write folder.
        f = open('case.json', 'w')
        # Dump the flowCart settings.
        json.dump(self.opts['RunControl'], f, indent=1)
        # Close the file.
        f.close()
        # Return to original location
        os.chdir(fpwd)
        
    # Write the PBS script.
    def WritePBS(self, i):
        """Write the PBS script(s) for a given case
        
        :Call:
            >>> fun3d.WritePBS(i)
        :Inputs:
            *fun3d*: :class:`pyFun.fun3d.Fun3d`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014-10-19 ``@ddalle``: First version
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
                fpbs = 'run_fun3d.%02i.pbs' % j
            else:
                # Use single PBS script with plain name.
                fpbs = 'run_fun3d.pbs'
            # Initialize the PBS script.
            f = open(fpbs, 'w')
            # Write the header.
            self.WritePBSHeader(f, i, j)
            
            # Initialize options to `run_FUN3D.py`
            flgs = ''

            # Simply call the advanced interface.
            f.write('\n# Call the FUN3D interface.\n')
            f.write('run_fun3d.py' + flgs)
            
            # Close the file.
            f.close()
        # Return.
        os.chdir(fpwd)
        
    # Call the correct :mod:`case` module to start a case
    def CaseStartCase(self):
        """Start a case by either submitting it or running it
        
        This function relies on :mod:`pyCart.case`, and so it is customized for
        the Cart3D solver only in that it calles the correct *case* module.
        
        :Call:
            >>> pbs = cart3d.CaseStartCase()
        :Inputs:
            *cart3d*: :class:`pyCart.cart3d.Cart3d`
                Instance of control class containing relevant parameters
        :Outputs:
            *pbs*: :class:`int` or ``None``
                PBS job ID if submitted successfully
        :Versions:
            * 2015-10-14 ``@ddalle``: First version
        """
        return case.StartCase()
        
        
# class Fun3d

