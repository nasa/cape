r"""
:mod:`cape.pylch.cntl`: Main Loci/CHEM run matrix controller
===============================================================

This module provides the :class:`Cntl` class that is specific to
``pylch``, the CAPE interface to Loci/CHEM.

"""

# Standard library
import os
import shutil

# Third-party
import numpy as np

# Local imports
from . import options
from . import casecntl
from . import databook
from .varsfile import VarsFile
from ..cfdx import cntl
from ..pyfun.mapbc import MapBC


# Primary class
class Cntl(cntl.Cntl):
  # === Class attributes ===
    # Names
    _solver = "fun3d"
    # Hooks to py{x} specific modules
    _case_mod = casecntl
    _databook_mod = databook
    # _report_mod = report
    # Hooks to py{x} specific classes
    _case_cls = casecntl.CaseRunner
    _opts_cls = options.Options
    # Other settings
    _fjson_default = "pyLCH.json"
    _warnmode_default = cntl.DEFAULT_WARNMODE

   # === Config ===
    def init_post(self):
        r"""Do ``__init__()`` actions specific to ``pylch``

        :Call:
            >>> cntl.init_post()
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                CAPE run matrix control instance
        :Versions:
            * 2024-10-17 ``@ddalle``: v1.0
        """
        # Read list of custom file control classes
        self.ReadVarsFile()
        self.ReadMapBC()
        self.ReadConfig()

  # === Case Preparation ===
    # Prepare a case
    @cntl.run_rootdir
    def PrepareCase(self, i: int):
        """Prepare case for running if necessary

        :Call:
            >>> cntl.PrepareCase(i)
        :Inputs:
            *cntl*: :class:`cape.pylava.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of case to analyze
        :Versions:
            * 2024-06-12 ``@sneuhoff``: v1.0
        """
        # Ensure case index is set
        self.opts.setx_i(i)
        # Get the existing status
        n = self.CheckCase(i)
        # Quit if prepared.
        if n is not None:
            return None
        # Get the run name.
        frun = self.x.GetFullFolderNames(i)
        # Case function
        self.CaseFunction(i)
        # Prepare the mesh (and create folders if necessary).
        self.PrepareMesh(i)
        # Go to case folder
        os.chdir(frun)
        # Write the conditions to a simple JSON file.
        self.x.WriteConditionsJSON(i)
        # Write a JSON file with contents of "RunControl" section
        self.WriteCaseJSON(i)
        # Write the PBS script.
        self.WritePBS(i)
        # Write .vars file
        self.PrepareVarsFile(i)

    @cntl.run_rootdir
    def PrepareVarsFile(self, i: int):
        r"""Prepare the run YAML file for each phase of one case

        :Call:
            >>> cntl.PrepareRunYAML(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        # Set case index for options
        self.opts.setx_i(i)
        # Set flight conditions
        self.PrepareRunYAMLFlightConditions(i)
        # Get user's selected file name
        project = self.GetProjectRootName(0)
        # Get name of case folder
        frun = self.x.GetFullFolderNames(i)
        # Enter said folder
        os.chdir(frun)
        # Loop through phases
        for j in self.opts.get_PhaseSequence():
            # Select file name
            varsfilename = f"{project}.{j:02d}.vars"
            # Other preparation
            ...
            # Write file
            self.VarsFile.write_yamlfile(varsfilename)

    # Prepare the flight conditions
    def PrepareVarsFileFlightConditions(self, i: int):
        r"""Prepare the flight conditions variables in a LAVA YAML file

        :Call:
            >>> cntl.PrepareRunYAMLFlightConditions(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        # Get properties
        mach = self.x.GetMach(i)
        rho = self.x.GetDensity(i, units="kg/m^3")
        T = self.x.GetTemperature(i, units="K")
        a = self.x.GetAlpha(i)
        b = self.x.GetBeta(i)
        # Get vars file interface
        opts = self.VarsFile
        # Set Mach number
        if mach is not None:
            opts.set_mach(mach)
        # Set angle of attack
        if a is not None:
            opts.set_alpha(a)
        # Set sideslip angle
        if b is not None:
            opts.set_beta(b)
        # Set density if specified
        if rho is not None:
            opts.set_density(rho)
        # Set temperature if specified
        if T is not None:
            opts.set_temperature(T)

   # === Input files and BCs ===
    # Get the project rootname
    def GetProjectRootName(self, j: int = 0) -> str:
        r"""Get the project root name

        This is taken directly from the main JSON file

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
            * 2024-10-17 ``@ddalle``: v1.0
        """
        return self.opts.get_ProjectName(j=j)

    # Read the namelist
    def ReadVarsFile(self, j: int = 0, q: bool = True):
        r"""Read the ``{project}.vars`` file

        :Call:
            >>> cntl.ReadVarsFile(j=0, q=True)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of the pyFun control class
            *j*: :class:`int`
                Phase number
            *q*: :class:`bool`
                Whether or not to read to *VarsFile*, else *VarsFile0*
        :Versions:
            * 2024-10-17 ``@ddalle``: v1.0
        """
        # Namelist file
        fvars = self.opts.get_VarsFile(j)
        # Check for empty value
        if fvars is None:
            return
        # Check for absolute path
        if not os.path.isabs(fvars):
            # Use path relative to JSON root
            fvars = os.path.join(self.RootDir, fvars)
        # Read the file
        vfile = VarsFile(fvars)
        # Save it.
        if q:
            # Read to main slot for modification
            self.VarsFile = vfile
        else:
            # Template for reading original parameters
            self.VarsFile0 = vfile

    # Read the boundary condition map
    @cntl.run_rootdir
    def ReadMapBC(self, j: int = 0, q: bool = True):
        r"""Read the FUN3D boundary condition map

        :Call:
            >>> cntl.ReadMapBC(q=True)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of the pyFun control class
            *q*: {``True``} | ``False``
                Whether or not to read to *MapBC*, else *MapBC0*
        :Versions:
            * 2016-03-30 ``@ddalle``: v1.0 (pyfun)
            * 2024-10-17 ``@ddalle``: v1.0
        """
        # MapBC file
        fmapbc = self.opts.get_MapBCFile(j)
        # Check if specified
        if fmapbc is None:
            return
        # Read the file
        bc = MapBC(self.opts.get_MapBCFile(j))
        # Save it.
        if q:
            # Read to main slot.
            self.MapBC = bc
        else:
            # Template
            self.MapBC0 = bc

   # === Mesh ===
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
            * 2015-10-19 ``@ddalle``: v1.0 (``pyfun``)
            * 2024-10-17 ``@ddalle``: v1.0
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
        # Starting phase
        phase0 = self.opts.get_PhaseSequence(0)
        # Project name
        fproj = self.GetProjectRootName(phase0)
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
                if not os.path.isabs(fbc):
                    fbc = os.path.join(self.RootDir, fbc)
                # Copy the file
                shutil.copyfile(fbc, '%s.aflr3bc' % fproj)
            # Surface configuration file
            fxml = self.opts.get_ConfigFile()
            # Write it if necessary
            if fxml:
                # Absolute file name
                if not os.path.isabs(fxml):
                    fxml = os.path.join(self.RootDir, fxml)
                # Copy the file
                shutil.copyfile(fxml, '%s.xml' % fproj)
            # Check intersection status.
            if self.opts.get_intersect():
                # Names of triangulation files
                fvtri = "%s.tri" % fproj
                fctri = "%s.c.tri" % fproj
                fftri = "%s.f.tri" % fproj
                # Write tri file as non-intersected; each volume is one CompID
                if not os.path.isfile(fvtri):
                    self.tri.WriteVolTri(fvtri)
                # Write the existing triangulation with existing CompIDs.
                if not os.path.isfile(fctri):
                    self.tri.WriteCompIDTri(fctri)
                # Write the farfield and source triangulation files
                if not os.path.isfile(fftri):
                    self.tri.WriteFarfieldTri(fftri)
            elif self.opts.get_verify():
                # Names of surface mesh files
                fitri = "%s.i.tri" % fproj
                fsurf = "%s.surf" % fproj
                # Write the tri file
                if not os.path.isfile(fitri):
                    self.tri.Write(fitri)
                # Write the AFLR3 surface file
                if not os.path.isfile(fsurf):
                    self.tri.WriteSurf(fsurf)
            else:
                # Names of surface mesh files
                fsurf = "%s.surf" % fproj
                # Write the AFLR3 surface file only
                if not os.path.isfile(fsurf):
                    self.tri.WriteSurf(fsurf)
       # --------------------
       # Volume mesh creation
       # --------------------
        # Get functions for mesh functions.
        keys = self.x.GetKeysByType('MeshFunction')
        # Loop through the mesh functions
        for key in keys:
            # Get the function for this *MeshFunction*
            func = self.x.defns[key]['Function']
            # Form args and kwargs
            a = (self, self.x[key][i])
            kw = dict(i=i)
            # Apply it
            self.exec_modfunction(func, a, kw, name="RunMatrixMeshFunction")

    # Get list of raw file names
    def GetInputMeshFileNames(self) -> list:
        r"""Return the list of mesh files from file

        :Call:
            >>> fname = cntl.GetInputMeshFileNames()
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
        :Outputs:
            *fname*: :class:`list`\ [:class:`str`]
                List of file names read from root directory
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0
        """
        # Get the file names from *opts*
        fname = self.opts.get_MeshFile()
        # Ensure list
        if fname is None:
            # Remove ``None``
            return []
        elif isinstance(fname, (list, np.ndarray, tuple)):
            # Return list-like as list
            return list(fname)
        else:
            # Convert to list.
            return [fname]

    # Get list of mesh file names that should be in a case folder
    def GetProcessedMeshFileNames(self) -> list:
        r"""Return the list of mesh files that are written

        :Call:
            >>> fname = cntl.GetProcessedMeshFileNames()
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE main control instance
        :Outputs:
            *fname*: :class:`list`\ [:class:`str`]
                List of file names written to case folders
        :Versions:
            * 2024-10-17 ``@ddalle``: v1.0
        """
        # Initialize output
        fname = []
        # Loop through input files.
        for fn in self.GetInputMeshFileNames():
            # Get processed name
            fname.append(self.ProcessMeshFileName(fn))
        # Output
        return fname

    # Process a mesh file name to use the project root name
    def ProcessMeshFileName(self, fname: str) -> str:
        r"""Return a mesh file name using the project root name

        :Call:
            >>> fout = cntl.ProcessMeshFileName(fname)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
            *fname*: :class:`str`
                Raw file name to be converted to case-folder file name
        :Outputs:
            *fout*: :class:`str`
                Name of file name using project name as prefix
        :Versions:
            * 2016-04-05 ``@ddalle``: v1.0 (pyfun)
            * 2024-10-17 ``@ddalle``: v1.0
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
