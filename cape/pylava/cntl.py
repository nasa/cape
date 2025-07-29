# -*- coding: utf-8 -*-
r"""
:mod:`cape.pylavacntl`: LAVA control module
==================================================

This module provides tools to quickly setup basic or complex LAVA
run matrices and serve as an executive for pre-processing, running,
post-processing, and managing the solutions. A collection of cases
combined into a run matrix can be loaded using the following commands.

    .. code-block:: pycon

        >>> import cape.pylava
        >>> cntl = cape.pylava.cntl.Cntl("pyLava.json")
        >>> cntl
        <cape.pylava.Cntl(nCase=907)>
        >>> cntl.x.GetFullFolderNames(0)
        'poweroff/m1.5a0.0b0.0'


An instance of this :class:`cape.pylava.cntl.Cntl` class has many
attributes, which include the run matrix (``cntl.x``), the options
interface (``cntl.opts``), and optionally the data book
(``cntl.DataBook``), the appropriate input files (such as
``cntl.Namelist``), and possibly others.

    ====================   =============================================
    Attribute              Class
    ====================   =============================================
    *cntl.x*               :class:`cape.runmatrix.RunMatrix`
    *cntl.opts*            :class:`cape.pylava.options.Options`
    *cntl.DataBook*        :class:`cape.pylava.databook.DataBook`
    *cntl.Namelist*        :class:`cape.pylava.namelist.Namelist`
    ====================   =============================================

Finally, the :class:`Cntl` class is subclassed from the
:class:`cape.cntl.Cntl` class, so any methods available to the CAPE
class are also available here.
"""

# Standard library
import os

# Third-party
import numpy as np

# Local imports
from . import options
from . import casecntl
from . import databook
from . import report
from .yamlfile import RunYAMLFile
from .runinpfile import CartInputFile
from ..optdict import OptionsDict
from ..cfdx import cntl as capecntl
from ..cfdx.cmdgen import infix_phase


# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyLavaFolder = os.path.split(_fname)[0]

# Constants
DEG = np.pi / 180.0


# Class to read input files
class Cntl(capecntl.Cntl):
    r"""Class for handling global options and setup for LAVA

    This class is intended to handle all settings used to describe a
    group of LAVA cases. For situations where it is not sufficiently
    customized, it can be used partially, e.g., to set up a Mach/alpha
    sweep for each single control variable setting.

    The settings are read from a yaml or json file.

    :Call:
        >>> cntl = cape.pylava.Cntl(fname="pyLava.yaml")
    :Inputs:
        *fname*: :class:`str`
            Name of cape.pylava input file
    :Outputs:
        *cntl*: :class:`cape.pylava.cntl.Cntl`
            Instance of the cape.pylava control class
    :Data members:
        *cntl.opts*: :class:`dict`
            Dictionary of options for this case (directly from *fname*)
        *cntl.x*: :class:`cape.pylava.runmatrix.RunMatrix`
            Values and definitions for variables in the run matrix
        *cntl.RootDir*: :class:`str`
            Absolute path to the root directory
    :Versions:
        * 2024-04-25 ``@sneuhoff``: v1.0
    """
  # === Class Attributes ===
    _name = "pylava"
    _solver = "lava"
    _databook_mod = databook
    _case_cls = casecntl.CaseRunner
    _opts_cls = options.Options
    _report_cls = report.Report
    _fjson_default = "pyLava.json"
    _tri_ext = "fro"
    yaml_default = "run_default.yaml"
    _zombie_files = (
        "*.out",
        "*.log")

  # === __DUNDER__ ===
    # Initialization method
    def __init__(self, fname=None):
        r"""Initialization method for :mod:`cape.pylava.cntl.Cntl`
        Handles yaml files for cape input by conversion from
        yaml to json.

        :Versions:
            * 2024-10-01 ``sneuhoff``: v1.0
        """
        # Check if json or yaml
        fext = None if fname is None else fname.split('.')[-1]
        # Check for YAML vs JSON
        if fext in ("yaml", "yml"):
            # Convert yaml to json
            fout = f"{fname}.json"
            inyaml = OptionsDict(fname)
            inyaml.write_jsonfile(fout)
            # Call parent init using json
            super().__init__(fout)
        else:
            # Assume it's json, call parent init
            super().__init__(fname)

  # === Init config ===
    def init_post(self):
        r"""Do ``__init__()`` actions specific to ``pylava``

        :Call:
            >>> cntl.init_post()
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
        :Versions:
            * 2024-10-09 ``@ddalle``: v1.0
        """
        # Get solver
        solver = self.opts.get_LAVASolver()
        # Read list of custom file control classes
        if solver == "curvilinear":
            self.ReadRunYAML()
        elif solver == "cartesian":
            self.ReadCartInputFile()

  # === Case Preparation ===
    # Prepare a case
    @capecntl.run_rootdir
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
        # Write YAML file
        self.PrepareInputFile(i)

    # Prepare the mesh for case *i* (if necessary)
    @capecntl.run_rootdir
    def PrepareMesh(self, i: int):
        r"""Copy/link mesh files into case folder

        :Call:
            >>> cntl.PrepareMesh(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix controller instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
            * 2025-07-15 ``@ddalle``: v2.0; map *LAVASolver* value
        """
        # Check solver type
        solver = self.opts.get_LAVASolver()
        # Filter
        if solver == "curvilinear":
            self.PrepareMeshCurvilinear(i)
        else:
            self.PrepareMeshUnstructured(i)

    # Prepare curvilinear mesh
    def PrepareMeshCurvilinear(self, i: int):
        r"""Copy/link mesh files into case folder

        :Call:
            >>> cntl.PrepareMeshCurvilinear(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix controller instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        self.prepare_mesh_overset(i)

  # === Input files ===
   # --- Main switch ---
    def PrepareInputFile(self, i: int):
        r"""Prepare main input file, depends on LAVA solver is in use

        :Call:
            >>> cntl.PrepareInputFile(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2025-07-15 ``@ddalle``: v1.0
        """
        # Get solver type
        solver = self.opts.get_LAVASolver()
        # Switch
        if solver == "curvilinear":
            self.PrepareRunYAML(i)
        elif solver == "cartesian":
            self.PrepareRunInputs(i)

   # --- run.yaml ---
    # Read template YAML file
    def ReadRunYAML(self):
        r"""Read run YAML file, using template if setting is empty

        :Call:
            >>> cntl.ReadRunYAML()
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
        :Versions:
            * 2024-08-19 ``@sneuhoff``: v1.0 (``ReadInputFile()``)
            * 2024-10-09 ``@ddalle``: v2.0
        """
        # Get name of file to read
        fname = self.opts.get_RunYAMLFile()
        # Check for template
        if fname is None:
            # Read template
            fabs = os.path.join(PyLavaFolder, "templates", "run.yaml")
        else:
            # Absolutize
            fabs = os.path.join(self.RootDir, fname)
        # Read it
        self.YamlFile = RunYAMLFile(fabs)

    @capecntl.run_rootdir
    def PrepareRunYAML(self, i: int):
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
        yamlbase = self.opts.get_lava_yamlfile()
        # Get name of case folder
        frun = self.x.GetFullFolderNames(i)
        # Enter said folder
        os.chdir(frun)
        # Loop through phases
        for j in self.opts.get_PhaseSequence():
            # Select file name
            if isinstance(yamlbase, list):
                # Specified by phase
                yamlfile = self.opts.get_lava_yamlfile(j)
            else:
                # Add phase infix
                yamlfile = infix_phase(yamlbase, j)
            # Other preparation
            ...
            # Write file
            self.YamlFile.write_yamlfile(yamlfile)

    # Prepare the flight conditions
    def PrepareRunYAMLFlightConditions(self, i: int):
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
        u = self.x.GetVelocity(i, units="m/s")
        p = self.x.GetPressure(i, units="Pa")
        T = self.x.GetTemperature(i, units="K")
        a = self.x.GetAlpha(i)
        b = self.x.GetBeta(i)
        # Get YAML interface
        opts = self.YamlFile
        # Set velocity if any velocity setting was given
        if u is not None:
            opts.set_umag(u)
        # Set angle of attack
        if a is not None:
            opts.set_alpha(a)
        # Set sideslip angle
        if b is not None:
            opts.set_beta(b)
        # Set pressure if specified
        if p is not None:
            opts.set_pressure(p)
        # Set temperature if specified
        if T is not None:
            opts.set_temperature(T)
        # Set nonlinear iterations
        np = int(self.opts.get_PhaseIters())
        opts.set_lava_subopt('nonlinearsolver', 'iterations', np)

   # --- run.inputs ---
    # Read template "run.input" file
    def ReadCartInputFile(self):
        r"""Read LAVA-Cartesian input file, ``run.inputs``

        :Call:
            >>> cntl.ReadCartInputFile()
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
        :Version:
            * 2025-07-14 ``@ddalle``: v1.0
        """
        # Get name of file to read
        fname = self.opts.get_CartInputFile()
        # Check for it
        if fname is None:
            # Use template
            fabs = os.path.join(PyLavaFolder, "templates", "run.inputs")
        elif not os.path.isabs(fname):
            # Absolutize
            fabs = os.path.join(self.RootDir, fname)
        else:
            # Already absolute
            fabs = fname
        # Read it if possible
        if os.path.isfile(fabs):
            self.CartInputs = CartInputFile(fabs)

    # Prepare "run.inputs" file
    @capecntl.run_rootdir
    def PrepareRunInputs(self, i: int):
        r"""Prepare the ``run.inputs`` file for each phase of one case

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
        self.PrepareRunInputsFlightConditions(i)
        # Get user's selected file name
        basename = "run.inputs"
        # Get name of case folder
        frun = self.x.GetFullFolderNames(i)
        # Enter said folder
        os.chdir(frun)
        # Loop through phases
        for j in self.opts.get_PhaseSequence():
            # Select file name for this phase
            runfile = infix_phase(basename, j)
            # Select arbitrary options for this phase in JSON
            jsonrunopts = self.opts.select_runinputs_phase(j)
            # Apply options from JSON
            runopts = self.CartInputs
            if runopts is not None:
                runopts.apply_dict(jsonrunopts)
            # Write file
            self.CartInputs.write(runfile)

    # Prepare flight conditions portion of "run.inputs"
    def PrepareRunInputsFlightConditions(self, i: int):
        r"""Prep the reference conditions section of LAVA-cart file

        :Call:
            >>> cntl.PrepareRunInputsFlightConditions(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2025-07-15 ``@ddalle``: v1.0
        """
        # Get properties
        uinf = self.x.GetVelocity(i, units="m/s")
        p = self.x.GetPressure(i, units="Pa")
        r = self.x.GetDensity(i, units="kg/m^3")
        t = self.x.GetTemperature(i, units="K")
        m = self.x.GetMach(i)
        a = self.x.GetAlpha(i)
        b = self.x.GetBeta(i)
        rey = self.x.GetReynoldsNumber(i, units="1/m")
        # Get YAML interface
        opts = self.CartInputs
        # Get current properties to see what user intended
        ui = opts.get_refcond("velocity")
        pi = opts.get_refcond("pressure")
        ri = opts.get_refcond("density")
        ti = opts.get_refcond("temperature")
        rei = opts.get_refcond("Re")
        # See which flags we are expecting; need at least two
        qre = rei is not None
        qr = (ri is not None)
        qp = (pi is not None) or not (qr or qre)
        qt = (ti is not None) or not (qr and qp) or qre
        # Check for density
        if qr and (r is not None):
            opts.set_density(r)
        # Check for temperature
        if qt and (t is not None):
            opts.set_temperature(t)
        # Check for pressure
        if qp and (p is not None):
            opts.set_pressure(p)
        if qre and (rey is not None):
            opts.set_refcond("Re", rey)
        # Check for velocity magnitude
        if ui is None:
            # Set Mach number
            if m is not None:
                opts.set_mach(m)
            # Set angles
            if a is not None:
                opts.set_alpha(a)
            if b is not None:
                opts.set_beta(b)
        else:
            # Calculate angle
            ca = np.cos(a*DEG)
            cb = np.cos(b*DEG)
            sa = np.sin(a*DEG)
            sb = np.sin(b*DEG)
            # Components
            u = uinf*ca*cb
            v = -uinf*sb
            w = uinf*sa*cb
            # Set velocity
            opts.set_refcond("velocity", [u, v, w])

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
        print("  Writing input files 0 to %s" % (nPhase-1))
        self.PrepareInputFile(i)
        # Write PBS scripts
        nPBS = self.opts.get_nPBS()
        print("  Writing PBS scripts 0 to %s" % (nPBS-1))
        self.WritePBS(i)
