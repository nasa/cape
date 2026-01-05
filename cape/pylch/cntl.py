r"""
:mod:`cape.pylch.cntl`: Main Loci/CHEM run matrix controller
===============================================================

This module provides the :class:`Cntl` class that is specific to
``pylch``, the CAPE interface to Loci/CHEM.

"""

# Standard library
import os

# Third-party

# Local imports
from . import options
from . import casecntl
from .varsfile import VarsFile
from ..cfdx import cntl
from ..filecntl.mapbcfile import MapBCFile


# Primary class
class Cntl(cntl.Cntl):
  # === Class attributes ===
    # Names
    _solver = "chem"
    # Hooks to py{x} specific modules
    # _report_cls = report.Report
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

    # Use unstructured mesh pre
    def PrepareMesh(self, i: int):
        """Prepare mesh using unstructured protocol

        :Call:
            >>> cntl.PrepareMesh(i)
        :Inputs:
            *cntl*: :class:`cape.pylava.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of case to analyze
        :Versions:
            * 2025-07-15 ``@ddalle``: v1.0
        """
        self.PrepareMeshUnstructured(i)

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
        # Declare boundary condition for each surf
        self.PrepareVarsFileBCs(i)
        # Set flight conditions
        self.PrepareVarsFileFlightConditions(i)
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
            # Get number of iterations to run
            niter = self.opts.get_nIter(j)
            # Other settings from JSON
            vopts = self.opts.select_vars_phase(j)
            # Set number of iterations
            self.VarsFile.set_iter(niter)
            # Other preparation
            self.VarsFile.apply_dict(vopts)
            # Write file
            self.VarsFile.write(varsfilename)

    # Prepare the flight conditions
    def PrepareVarsFileFlightConditions(self, i: int):
        r"""Prepare the flight conditions in a Loci/CHEM ``.vars`` file

        :Call:
            >>> cntl.PrepareVarsFileFlightConditions(i)
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
            opts.set_rho(rho)
        # Set temperature if specified
        if T is not None:
            opts.set_temperature(T)

    # Prepare boundary conditions
    def PrepareVarsFileBCs(self, i: int):
        r"""PRepare the boundary conditions in Loci/CHEM ``.vars`` file

        :Call:
            >>> cntl.PrepareVarsFileBCs(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2025-05-09 ``@ddalle``: v1.0
        """
        # Check for MapBC
        mapbc = getattr(self, "MapBC", None)
        if mapbc is None:
            return
        # Get vars file interface
        opts = self.VarsFile
        # Loop through entries of ``.mapbc`` file
        for j, bc in enumerate(self.MapBC.BCs):
            # Get component name and number
            comp = self.MapBC.Names[j]
            compid = self.MapBC.CompID[j]
            # Formualte raw name
            rawcomp = f"BC_{compid}"
            # Check boundary condition
            if bc == 4000:
                opts.set_comp_wall(comp)
                opts.set_comp_wall(rawcomp)
            else:
                opts.set_comp_farfield(comp)
                opts.set_comp_farfield(rawcomp)

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
        bc = MapBCFile(self.opts.get_MapBCFile(j))
        # Save it.
        if q:
            # Read to main slot.
            self.MapBC = bc
        else:
            # Template
            self.MapBC0 = bc

   # === Mesh ===
